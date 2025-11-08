<template>
  <div id="app">
    <el-config-provider :locale="locale">
      <!-- 导航栏 -->
      <AppHeader v-if="showHeader" />
      
      <!-- 主内容区域 -->
      <main :class="mainContentClass">
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
import { computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import { useLoadingStore } from '@/stores/loading'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

const loadingStore = useLoadingStore()
const locale = zhCn
const route = useRoute()

const showHeader = computed(() => route.meta?.layout !== 'auth')
const mainContentClass = computed(() => ({
  'main-content': true,
  'main-content--auth': !showHeader.value
}))

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
  background: #f5f5f5;
  min-height: 100vh;
}

.main-content {
  padding-top: 72px;
  min-height: calc(100vh - 72px);
}

.main-content--auth {
  padding-top: 0;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

// Element Plus 样式覆盖
.el-loading-mask {
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
}

.el-button {
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

  &--primary {
    background: #409EFF !important;
    background-color: #409EFF !important;
    background-image: none !important;
    border-color: #409EFF;

    &:hover {
      background: #66b1ff !important;
      background-color: #66b1ff !important;
      background-image: none !important;
      border-color: #66b1ff;
    }

    &:active {
      background: #3a8ee6 !important;
      background-color: #3a8ee6 !important;
      background-image: none !important;
      border-color: #3a8ee6;
    }
  }

  &--small {
    &.el-button--primary {
      background: #409EFF !important;
      background-color: #409EFF !important;
      background-image: none !important;
    }
  }
}

.el-card {
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  background: #ffffff;
}

.el-input__wrapper {
  border-radius: 8px;
  transition: all 0.3s ease;

  &:hover {
    border-color: #409EFF;
  }

  &.is-focus {
    border-color: #409EFF;
    box-shadow: 0 0 0 1px rgba(64, 158, 255, 0.2);
  }
}

.el-select .el-input__wrapper {
  border-radius: 8px;

  &:hover {
    border-color: #409EFF;
  }

  &.is-focus {
    border-color: #409EFF;
  }
}

.el-tag {
  border-radius: 6px;

  &--primary {
    background-color: rgba(64, 158, 255, 0.1);
    border-color: #409EFF;
    color: #409EFF;
  }
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
