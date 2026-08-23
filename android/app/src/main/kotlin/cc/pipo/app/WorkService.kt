package cc.pipo.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder

/**
 * 处理期间把这个进程钉在前台档。
 *
 * 为什么需要
 * ----------
 * 分析加剪辑跑在 Activity 的协程里。实测（模拟器，AOSP）切到后台之后
 * **活儿确实还在跑** —— 60 秒 CPU tick 涨了 3663，比前台还快，因为不用
 * 渲染界面。但同一时刻 `oom_score_adj` 从 0 跳到 **900**，也就是
 * 「缓存应用」档：内存一紧张就是第一批被杀，而**被杀时不抛异常、
 * 没有任何提示**，用户看到的就是「回来一看什么都没了」。
 * 模拟器有 16GB 所以扛过去了，手机不一定 —— EMUI 尤其激进。
 *
 * 界面上已经写了「请留在这一页」，但那是把系统的问题推给用户。
 * 前台服务才是真解：进程优先级不再随着切走而掉下去。
 *
 * 只管优先级，不搬代码
 * --------------------
 * **活儿仍然留在 Activity 的协程里，这个服务什么都不做。**
 * oom_score_adj 是**按进程**算的，起一个前台服务就足以把整个进程抬上去；
 * 为了这个把整条流水线搬进 Service，要多出一套跨进程的进度回传和
 * 生命周期管理，而收益是零。
 *
 * 类型选择
 * --------
 * API 34 起前台服务必须声明类型。`mediaProcessing` 正是为「转码媒体文件」
 * 定义的，语义最贴合；它在 API 34 才有，更早的版本不需要类型。
 * 注意它有每天 6 小时的配额 —— 我们一次最多二十来分钟，够用。
 */
class WorkService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val n = build(this)
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING)
        } else {
            startForeground(ID, n)
        }
        // 被系统杀掉之后不要自己重来：那时候 Activity 那边的协程早没了，
        // 重启一个空转的服务只会挂一个永远不动的通知。
        return START_NOT_STICKY
    }

    private fun build(ctx: Context): Notification {
        val nm = ctx.getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // IMPORTANCE_LOW：不响铃不震动。这条通知的作用是「让系统知道
            // 我们在干正事」，不是提醒用户 —— 用户就在应用里看着进度条。
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL, "剪辑进行中", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            ctx, 0, Intent(ctx, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return Notification.Builder(ctx, CHANNEL)
            .setContentTitle("Pipo 正在剪辑")
            .setContentText("完成前请不要清理后台")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val CHANNEL = "pipo_work"
        private const val ID = 1

        /**
         * 开工。**失败不能影响主流程** —— 前台服务受配额和厂商策略限制，
         * 起不来时最坏也只是回到从前（切后台可能被杀），
         * 不该让它把一次本来能成的剪辑变成一个崩溃。
         */
        fun start(ctx: Context) {
            runCatching {
                val i = Intent(ctx, WorkService::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    ctx.startForegroundService(i)
                } else {
                    ctx.startService(i)
                }
            }
        }

        /** 收工。成功、报错、用户放弃都要调 —— 否则通知会一直挂着。 */
        fun stop(ctx: Context) {
            runCatching { ctx.stopService(Intent(ctx, WorkService::class.java)) }
        }
    }
}
