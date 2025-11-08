import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import { useAuthStore } from '@/stores/auth'

// Element Plus
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// 全局样式
import './styles/main.scss'

const app = createApp(App)
const pinia = createPinia()

// 全局错误处理器 - 防止错误导致页面崩溃
app.config.errorHandler = (err, instance, info) => {
  console.error('全局错误捕获:', err)
  console.error('错误信息:', info)
  console.error('组件实例:', instance)
  // 不抛出错误，避免页面崩溃
}

// 处理未捕获的 Promise 错误
window.addEventListener('unhandledrejection', event => {
  console.error('未处理的 Promise 错误:', event.reason)
  event.preventDefault() // 阻止默认行为（页面刷新）
})

// 注册Element Plus图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.use(pinia)

const authStore = useAuthStore()
await authStore.initialize()

app.use(router)
app.use(ElementPlus)

app.mount('#app')
