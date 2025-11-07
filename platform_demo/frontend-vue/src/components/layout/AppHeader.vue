<template>
  <header class="app-header">
    <div class="header-container">
      <!-- Logo和标题 -->
      <div class="header-left">
        <div class="logo">
          <el-icon size="32" color="#ffffff">
            <Odometer />
          </el-icon>
        </div>
        <h1 class="app-title">水质监测平台</h1>
      </div>

      <!-- 导航菜单 -->
      <nav class="header-nav">
        <el-menu
          :default-active="$route.path"
          mode="horizontal"
          :ellipsis="false"
          @select="handleMenuSelect"
        >
          <el-menu-item index="/">
            <el-icon><DataLine /></el-icon>
            <span>仪表板</span>
          </el-menu-item>

          <el-menu-item index="/monitoring">
            <el-icon><View /></el-icon>
            <span>实时监测</span>
          </el-menu-item>

          <el-menu-item index="/prediction">
            <el-icon><TrendCharts /></el-icon>
            <span>智能预测</span>
          </el-menu-item>

          <el-menu-item index="/analysis">
            <el-icon><PieChart /></el-icon>
            <span>数据分析</span>
          </el-menu-item>
          
          <el-menu-item index="/quality">
            <el-icon><CircleCheck /></el-icon>
            <span>数据质量</span>
          </el-menu-item>
          
          <el-menu-item index="/spatial">
            <el-icon><Location /></el-icon>
            <span>空间分析</span>
          </el-menu-item>
        </el-menu>
      </nav>

      <!-- 右侧工具栏 -->
      <div class="header-right">
        <!-- 全屏切换 -->
        <el-button
          text
          @click="toggleFullscreen"
          class="header-btn"
        >
          <el-icon size="20">
            <FullScreen />
          </el-icon>
        </el-button>

        <!-- 刷新按钮 -->
        <el-button
          text
          @click="refreshPage"
          class="header-btn"
        >
          <el-icon size="20">
            <Refresh />
          </el-icon>
        </el-button>

        <!-- 设置按钮 -->
        <el-button
          text
          @click="openSettings"
          class="header-btn"
        >
          <el-icon size="20">
            <Setting />
          </el-icon>
        </el-button>

        <!-- 用户信息 -->
        <el-dropdown @command="handleUserAction">
          <div class="user-avatar">
            <el-avatar :size="32" :src="userAvatar">
              <el-icon><User /></el-icon>
            </el-avatar>
          </div>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="profile">
                <el-icon><User /></el-icon>
                个人资料
              </el-dropdown-item>
              <el-dropdown-item command="settings">
                <el-icon><Setting /></el-icon>
                设置
              </el-dropdown-item>
              <el-dropdown-item divided command="logout">
                <el-icon><SwitchButton /></el-icon>
                退出登录
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </div>
  </header>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  Odometer,
  DataLine,
  View,
  TrendCharts,
  PieChart,
  CircleCheck,
  Location,
  FullScreen,
  Refresh,
  Setting,
  User,
  SwitchButton
} from '@element-plus/icons-vue'

const router = useRouter()

const userAvatar = '' // 可以从用户store获取

// 菜单选择处理
const handleMenuSelect = (index) => {
  router.push(index)
}

// 全屏切换
const toggleFullscreen = () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen()
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen()
    }
  }
}

// 刷新页面
const refreshPage = () => {
  window.location.reload()
}

// 打开设置
const openSettings = () => {
  ElMessage.info('设置功能开发中...')
}

// 用户操作处理
const handleUserAction = (command) => {
  switch (command) {
    case 'profile':
      ElMessage.info('个人资料功能开发中...')
      break
    case 'settings':
      openSettings()
      break
    case 'logout':
      ElMessage.success('已退出登录')
      break
  }
}
</script>

<style lang="scss" scoped>
.app-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 72px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  z-index: $z-fixed;
  transition: all $transition-base;
}

.header-container {
  max-width: 1400px;
  margin: 0 auto;
  height: 100%;
  padding: 0 $spacing-6;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-left {
  display: flex;
  align-items: center;
  gap: $spacing-3;
  flex-shrink: 0;
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: $border-radius-lg;
  box-shadow: $shadow-md;
}

.app-title {
  font-size: $font-size-xl;
  font-weight: 700;
  color: $gray-800;
  margin: 0;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-nav {
  flex: 1;
  display: flex;
  justify-content: center;
  overflow-x: auto;
  
  // 隐藏滚动条但保持滚动功能
  &::-webkit-scrollbar {
    display: none;
  }
  
  -ms-overflow-style: none;
  scrollbar-width: none;
}

:deep(.el-menu--horizontal) {
  border-bottom: none;
  background: transparent;
  
  .el-menu-item {
    border-bottom: 2px solid transparent;
    transition: all $transition-base;
    
    &:hover {
      background: rgba(102, 126, 234, 0.1);
      color: #667eea;
    }
    
    &.is-active {
      background: rgba(102, 126, 234, 0.1);
      color: #667eea;
      border-bottom-color: #667eea;
    }
    
    .el-icon {
      margin-right: $spacing-2;
    }
  }
}

.header-right {
  display: flex;
  align-items: center;
  gap: $spacing-2;
  flex-shrink: 0;
}

.header-btn {
  width: 40px;
  height: 40px;
  border-radius: $border-radius-lg;
  transition: all $transition-base;
  
  &:hover {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
  }
}

.user-avatar {
  cursor: pointer;
  transition: all $transition-base;
  
  &:hover {
    transform: scale(1.05);
  }
}

// 响应式设计
@media (max-width: $breakpoint-lg) {
  .header-container {
    padding: 0 $spacing-4;
  }
  
  .app-title {
    font-size: $font-size-lg;
  }
}

@media (max-width: $breakpoint-md) {
  .header-nav {
    order: 3;
    flex-basis: 100%;
    margin-top: $spacing-4;
    justify-content: flex-start;
  }
  
  .header-container {
    flex-wrap: wrap;
    align-items: flex-start;
    padding-top: $spacing-4;
    height: auto;
    min-height: 72px;
  }
  
  .app-header {
    height: auto;
    min-height: 72px;
  }
}

@media (max-width: $breakpoint-sm) {
  .header-left {
    gap: $spacing-2;
  }
  
  .logo {
    width: 40px;
    height: 40px;
  }
  
  .app-title {
    font-size: $font-size-base;
  }
  
  :deep(.el-menu--horizontal) {
    .el-menu-item {
      padding: 0 $spacing-3;
      font-size: $font-size-sm;
      
      span {
        display: none;
      }
      
      .el-icon {
        margin-right: 0;
      }
    }
  }
}
</style>
