<template>
  <div id="app">
    <el-config-provider :locale="locale">
      <!-- 导航栏 -->
      <AppHeader />
      
      <!-- 主内容区域 -->
      <main class="main-content">
        <router-view />
      </main>
      
      <!-- 全局加载遮罩 -->
      <el-loading 
        v-if="loadingStore.globalLoading" 
        :fullscreen="true"
        text="加载中..."
      />
    </el-config-provider>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import AppHeader from '@/components/layout/AppHeader.vue'
import { useLoadingStore } from '@/stores/loading'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

const loadingStore = useLoadingStore()
const locale = zhCn

onMounted(() => {
  // 初始化应用
  console.log('🚀 水质监测平台已启动')
})
</script>

<style lang="scss">
#app {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.main-content {
  padding-top: 72px;
  min-height: calc(100vh - 72px);
}

// Element Plus 样式覆盖
.el-loading-mask {
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
}

.el-button {
  border-radius: 12px;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.el-card {
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(20px);
  background: rgba(255, 255, 255, 0.95);
}

.el-input__wrapper {
  border-radius: 12px;
  transition: all 0.3s ease;
}

.el-select .el-input__wrapper {
  border-radius: 12px;
}

// 平滑滚动
html {
  scroll-behavior: smooth;
}

// 自定义滚动条
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f5f9;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #94a3b8;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #64748b;
}
</style>
