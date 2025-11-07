import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Dashboard',
      component: () => import('@/views/Dashboard.vue'),
      meta: {
        title: '仪表板',
        icon: 'DataBoard'
      }
    },
    {
      path: '/monitoring',
      name: 'Monitoring',
      component: () => import('@/views/Monitoring.vue'),
      meta: {
        title: '实时监测',
        icon: 'Monitor'
      }
    },
    {
      path: '/prediction',
      name: 'Prediction',
      component: () => import('@/views/Prediction.vue'),
      meta: {
        title: '智能预测',
        icon: 'TrendCharts'
      }
    },
    {
      path: '/analysis',
      name: 'Analysis',
      component: () => import('@/views/Analysis.vue'),
      meta: {
        title: '数据分析',
        icon: 'DataAnalysis'
      }
    },
    {
      path: '/quality',
      name: 'Quality',
      component: () => import('@/views/Quality.vue'),
      meta: {
        title: '数据质量',
        icon: 'CircleCheck'
      }
    },
    {
      path: '/spatial',
      name: 'Spatial',
      component: () => import('@/views/Spatial.vue'),
      meta: {
        title: '空间分析',
        icon: 'Location'
      }
    }
  ]
})

// 路由守卫
router.beforeEach((to, from, next) => {
  // 设置页面标题
  if (to.meta?.title) {
    document.title = `${to.meta.title} - 水质监测平台`
  }
  
  next()
})

export default router
