<template>
  <div class="auth-wrapper">
    <div class="auth-background">
      <div class="auth-gradient"></div>
      <div class="auth-circle auth-circle--lg"></div>
      <div class="auth-circle auth-circle--sm"></div>
    </div>
    <div class="auth-container">
      <el-card class="auth-card" shadow="never">
        <div class="auth-card__left">
          <div class="brand">
            <h2>水质监测平台</h2>
            <p>用数据洞察水环境，辅助科学决策</p>
          </div>
          <ul class="feature-list">
            <li>
              <span class="dot"></span>
              实时掌握全国水质监测动态
            </li>
            <li>
              <span class="dot"></span>
              多模型预测分析，洞察趋势
            </li>
            <li>
              <span class="dot"></span>
              智能告警系统，及时发现异常
            </li>
          </ul>
        </div>
        <div class="auth-card__right">
          <div class="form-header">
            <h3>{{ activeTab === 'login' ? '欢迎回来' : '注册新账号' }}</h3>
            <p>{{ activeTab === 'login' ? '请登录后继续探索水质数据' : '加入平台，开始智能水质监测之旅' }}</p>
          </div>
          <el-tabs v-model="activeTab" stretch class="auth-tabs">
            <el-tab-pane label="账号登录" name="login">
              <el-form
                ref="loginFormRef"
                :model="loginForm"
                :rules="loginRules"
                label-position="top"
                @keyup.enter.native="handleLogin"
              >
                <el-form-item label="用户名" prop="username">
                  <el-input
                    v-model="loginForm.username"
                    placeholder="请输入用户名"
                    autocomplete="username"
                    prefix-icon="User"
                  />
                </el-form-item>
                <el-form-item label="密码" prop="password">
                  <el-input
                    v-model="loginForm.password"
                    type="password"
                    placeholder="请输入密码"
                    show-password
                    autocomplete="current-password"
                    prefix-icon="Lock"
                  />
                </el-form-item>
                <div class="form-actions">
                  <el-checkbox v-model="rememberMe">记住我</el-checkbox>
                  <el-button link type="primary" disabled>忘记密码？</el-button>
                </div>
                <el-form-item>
                  <el-button type="primary" :loading="authLoading" @click="handleLogin" class="auth-submit">
                    登录
                  </el-button>
                </el-form-item>
              </el-form>
            </el-tab-pane>
            <el-tab-pane label="快速注册" name="register">
              <el-form
                ref="registerFormRef"
                :model="registerForm"
                :rules="registerRules"
                label-position="top"
              >
                <el-form-item label="用户名" prop="username">
                  <el-input v-model="registerForm.username" placeholder="请输入用户名" autocomplete="username" />
                </el-form-item>
                <el-form-item label="密码" prop="password">
                  <el-input v-model="registerForm.password" type="password" placeholder="请输入密码" show-password autocomplete="new-password" />
                </el-form-item>
                <el-form-item label="确认密码" prop="confirmPassword">
                  <el-input v-model="registerForm.confirmPassword" type="password" placeholder="请再次输入密码" show-password autocomplete="new-password" />
                </el-form-item>
                <el-form-item label="邮箱（可选）" prop="email">
                  <el-input v-model="registerForm.email" placeholder="请输入邮箱" autocomplete="email" />
                </el-form-item>
                <el-form-item label="姓名（可选）" prop="full_name">
                  <el-input v-model="registerForm.full_name" placeholder="请输入姓名" />
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" :loading="authLoading" @click="handleRegister" class="auth-submit">
                    注册并登录
                  </el-button>
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const { loading } = authStore

const activeTab = ref('login')
const rememberMe = ref(false)
const authLoading = computed(() => loading.value)

const loginFormRef = ref(null)
const loginForm = reactive({
  username: '',
  password: ''
})

const loginRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' }
  ]
}

const registerFormRef = ref(null)
const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: '',
  email: '',
  full_name: ''
})

const registerRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, message: '用户名不少于3个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码不少于6个字符', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请再次输入密码', trigger: 'blur' },
    {
      validator: (_, value, callback) => {
        if (value !== registerForm.password) {
          callback(new Error('两次输入的密码不一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ],
  email: [
    { type: 'email', message: '邮箱格式不正确', trigger: 'blur' }
  ]
}

const redirectAfterLogin = () => {
  const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
  router.replace(redirect)
}

const handleLogin = () => {
  loginFormRef.value?.validate(async (valid) => {
    if (!valid) return
    const success = await authStore.login(
      {
        username: loginForm.username,
        password: loginForm.password
      },
      {
        remember: rememberMe.value
      }
    )
    if (success) {
      redirectAfterLogin()
    }
  })
}

const handleRegister = () => {
  registerFormRef.value?.validate(async (valid) => {
    if (!valid) return
    const success = await authStore.register(
      {
        username: registerForm.username,
        password: registerForm.password,
        email: registerForm.email,
        full_name: registerForm.full_name
      },
      {
        remember: rememberMe.value
      }
    )
    if (success) {
      redirectAfterLogin()
    }
  })
}
</script>

<style scoped lang="scss">
.auth-wrapper {
  position: relative;
  width: 100%;
  min-height: 100vh;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 60%, #f5f5f5 100%);
}

.auth-background {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
}

.auth-gradient {
  position: absolute;
  inset: 0;
  background: radial-gradient(120% 120% at 0% 0%, rgba(64, 158, 255, 0.2) 0%, rgba(64, 158, 255, 0) 45%),
    radial-gradient(120% 120% at 100% 0%, rgba(102, 177, 255, 0.25) 0%, rgba(102, 177, 255, 0) 40%),
    radial-gradient(150% 150% at 50% 100%, rgba(64, 158, 255, 0.12) 0%, rgba(64, 158, 255, 0) 55%);
}

.auth-circle {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.45);
  box-shadow: 0 20px 60px rgba(64, 158, 255, 0.15);

  &--lg {
    width: 320px;
    height: 320px;
    bottom: -120px;
    right: -60px;
  }

  &--sm {
    width: 180px;
    height: 180px;
    top: 60px;
    left: -40px;
  }
}

.auth-container {
  position: relative;
  width: min(980px, 92%);
}

.auth-card {
  border-radius: 28px;
  backdrop-filter: blur(16px);
  background: rgba(255, 255, 255, 0.92);
  box-shadow:
    0 40px 70px -40px rgba(15, 29, 84, 0.3),
    0 10px 40px rgba(64, 158, 255, 0.12);

  :deep(.el-card__body) {
    display: flex;
    padding: 0;
  }
}

.auth-card__left {
  flex: 0 0 44%;
  padding: $spacing-8;
  background: linear-gradient(140deg, #409eff 0%, #66b1ff 65%, #7dc4ff 100%);
  color: #fff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: $spacing-6;
  position: relative;
  overflow: hidden;

  &::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.18), transparent 55%);
    pointer-events: none;
  }
}

.brand {
  position: relative;
  z-index: 1;

  h2 {
    font-size: 36px;
    margin-bottom: $spacing-3;
    font-weight: 600;
  }

  p {
    font-size: 16px;
    opacity: 0.82;
    line-height: 1.7;
    margin: 0;
  }
}

.feature-list {
  position: relative;
  z-index: 1;
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: $spacing-3;

  li {
    display: flex;
    align-items: center;
    gap: $spacing-3;
    font-size: 15px;
    line-height: 1.6;

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.85);
      box-shadow: 0 0 0 6px rgba(255, 255, 255, 0.16);
    }
  }
}

.auth-card__right {
  flex: 1;
  padding: $spacing-8;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.form-header {
  margin-bottom: $spacing-5;

  h3 {
    margin: 0;
    font-size: 28px;
    color: #1f2d3d;
    font-weight: 600;
  }

  p {
    margin-top: $spacing-2;
    margin-bottom: 0;
    color: #909399;
    font-size: 14px;
  }
}

.auth-tabs {
  :deep(.el-tabs__nav-wrap::after) {
    display: none;
  }

  :deep(.el-tabs__active-bar) {
    height: 4px;
    border-radius: 4px;
  }

  :deep(.el-tabs__item) {
    font-weight: 500;
    padding: 0 20px;
  }
}

.form-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: $spacing-3;

  .el-button {
    padding: 0;
  }
}

.auth-submit {
  width: 100%;
  height: 44px;
  font-size: 15px;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(64, 158, 255, 0.25);
}

@media (max-width: 980px) {
  .auth-container {
    width: min(720px, 94%);
  }

  .auth-card__left {
    padding: $spacing-6;
  }
}

@media (max-width: $breakpoint-lg) {
  .auth-card {
    :deep(.el-card__body) {
      flex-direction: column;
    }
  }

  .auth-card__left {
    flex: none;
    border-radius: 28px 28px 0 0;
    text-align: center;
    align-items: center;
    gap: $spacing-4;
  }

  .feature-list li {
    justify-content: center;
  }

  .auth-card__right {
    padding: $spacing-6;
  }
}

@media (max-width: $breakpoint-sm) {
  .auth-wrapper {
    padding: $spacing-4 0;
  }

  .auth-container {
    width: 92%;
  }

  .auth-card__left {
    padding: $spacing-5;

    h2 {
      font-size: 28px;
    }
  }

  .auth-card__right {
    padding: $spacing-5;
  }
}
</style>

