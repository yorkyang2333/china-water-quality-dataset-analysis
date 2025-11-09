import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { authApi } from '@/api/auth'

const TOKEN_STORAGE_KEY = 'wqp_auth_token'
const TOKEN_SESSION_KEY = 'wqp_auth_token_session'
let initializationTask = null

export const useAuthStore = defineStore('auth', () => {
  const token = ref('')
  const user = ref(null)
  const loading = ref(false)
  const initialized = ref(false)

  const isAuthenticated = computed(() => !!token.value)

  const setAuthState = (newToken, newUser, persist = true) => {
    token.value = newToken || ''
    user.value = newUser || null

    // 清除所有之前的 token 存储
    localStorage.removeItem(TOKEN_STORAGE_KEY)
    sessionStorage.removeItem(TOKEN_SESSION_KEY)

    if (token.value) {
      // 总是在 sessionStorage 中存储（维持当前会话）
      sessionStorage.setItem(TOKEN_SESSION_KEY, token.value)

      // 如果 persist 为 true，也在 localStorage 中存储（长期保存）
      if (persist) {
        localStorage.setItem(TOKEN_STORAGE_KEY, token.value)
      }
    }
  }

  const initialize = async () => {
    if (initialized.value && !initializationTask) return
    if (initializationTask) {
      await initializationTask
      return
    }

    initializationTask = (async () => {
      // 优先从 localStorage 读取（已勾选"记住我"）
      let storedToken = localStorage.getItem(TOKEN_STORAGE_KEY)

      // 如果 localStorage 中没有，尝试从 sessionStorage 读取（未勾选但在当前会话中）
      if (!storedToken) {
        storedToken = sessionStorage.getItem(TOKEN_SESSION_KEY)
      }

      if (storedToken) {
        token.value = storedToken
        try {
          const { data } = await authApi.getProfile()
          if (data.success) {
            user.value = data.data
          } else {
            setAuthState('', null)
          }
        } catch (error) {
          console.warn('自动登录失败', error)
          setAuthState('', null)
        }
      }
      initialized.value = true
      initializationTask = null
    })()

    await initializationTask
  }

  const login = async (credentials, options = {}) => {
    loading.value = true
    try {
      const { data } = await authApi.login(credentials)
      if (data.success) {
        const persist = options.remember !== false
        setAuthState(data.data.token, data.data.user, persist)
        ElMessage.success('登录成功')
        return true
      }
      ElMessage.error(data.error || '登录失败')
      return false
    } catch (error) {
      ElMessage.error(error.response?.data?.error || '登录失败')
      return false
    } finally {
      loading.value = false
    }
  }

  const register = async (payload, options = {}) => {
    loading.value = true
    try {
      const { data } = await authApi.register(payload)
      if (data.success) {
        const persist = options.remember !== false
        setAuthState(data.data.token, data.data.user, persist)
        ElMessage.success('注册成功')
        return true
      }
      ElMessage.error(data.error || '注册失败')
      return false
    } catch (error) {
      ElMessage.error(error.response?.data?.error || '注册失败')
      return false
    } finally {
      loading.value = false
    }
  }

  const fetchProfile = async () => {
    if (!token.value) return null
    try {
      const { data } = await authApi.getProfile()
      if (data.success) {
        user.value = data.data
        return data.data
      }
      return null
    } catch (error) {
      console.error('获取用户信息失败', error)
      return null
    }
  }

  const updateProfile = async (payload) => {
    loading.value = true
    try {
      const { data } = await authApi.updateProfile(payload)
      if (data.success) {
        user.value = data.data
        ElMessage.success('资料已更新')
        return true
      }
      ElMessage.error(data.error || '更新失败')
      return false
    } catch (error) {
      ElMessage.error(error.response?.data?.error || '更新失败')
      return false
    } finally {
      loading.value = false
    }
  }

  const logout = (showMessage = true) => {
    // 清除所有 token 存储
    setAuthState('', null)
    if (showMessage) {
      ElMessage.success('已退出登录')
    }
  }

  return {
    token,
    user,
    loading,
    initialized,
    isAuthenticated,
    initialize,
    login,
    register,
    fetchProfile,
    updateProfile,
    logout
  }
})

