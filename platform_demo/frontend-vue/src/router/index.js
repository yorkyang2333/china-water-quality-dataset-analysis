import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: {
      title: '登录',
      layout: 'auth',
      public: true
    }
  },
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue'),
    meta: {
      title: '仪表板',
      icon: 'DataBoard',
      requiresAuth: true
    }
  },
  {
    path: '/monitoring',
    name: 'Monitoring',
    component: () => import('@/views/Monitoring.vue'),
    meta: {
      title: '实时监测',
      icon: 'Monitor',
      requiresAuth: true
    }
  },
  {
    path: '/prediction',
    name: 'Prediction',
    component: () => import('@/views/Prediction.vue'),
    meta: {
      title: '智能预测',
      icon: 'TrendCharts',
      requiresAuth: true
    }
  },
  {
    path: '/analysis',
    name: 'Analysis',
    component: () => import('@/views/Analysis.vue'),
    meta: {
      title: '数据分析',
      icon: 'DataAnalysis',
      requiresAuth: true
    }
  },
  {
    path: '/quality',
    name: 'Quality',
    component: () => import('@/views/Quality.vue'),
    meta: {
      title: '数据质量',
      icon: 'CircleCheck',
      requiresAuth: true
    }
  },
  {
    path: '/spatial',
    name: 'Spatial',
    component: () => import('@/views/Spatial.vue'),
    meta: {
      title: '空间分析',
      icon: 'Location',
      requiresAuth: true
    }
  },
  {
    path: '/profile',
    name: 'Profile',
    component: () => import('@/views/Profile.vue'),
    meta: {
      title: '个人资料',
      icon: 'User',
      requiresAuth: true
    }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  await authStore.initialize()

  if (to.meta?.requiresAuth && !authStore.isAuthenticated) {
    next({
      name: 'Login',
      query: { redirect: to.fullPath }
    })
    return
  }

  if (to.name === 'Login' && authStore.isAuthenticated) {
    const redirect = typeof to.query.redirect === 'string' ? to.query.redirect : '/'
    next(redirect)
    return
  }

  if (to.meta?.title) {
    document.title = `${to.meta.title} - 水质监测平台`
  }

  next()
})

export default router
