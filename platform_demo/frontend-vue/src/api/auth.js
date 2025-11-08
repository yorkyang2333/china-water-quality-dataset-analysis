import api from '@/api/waterQuality'

export const authApi = {
  login(payload) {
    return api.post('/auth/login', payload)
  },
  register(payload) {
    return api.post('/auth/register', payload)
  },
  getProfile() {
    return api.get('/user/profile')
  },
  updateProfile(payload) {
    return api.put('/user/profile', payload)
  }
}

export default authApi

