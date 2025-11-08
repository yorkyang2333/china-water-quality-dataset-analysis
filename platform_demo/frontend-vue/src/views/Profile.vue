<template>
  <div class="profile-page">
    <el-row :gutter="24">
      <el-col :span="8" :xs="24">
        <el-card shadow="never" class="profile-card">
          <div class="profile-avatar">
            <el-avatar :size="96" :src="form.avatar_url">
              <el-icon><UserFilled /></el-icon>
            </el-avatar>
            <h2>{{ displayName }}</h2>
            <p class="username">@{{ form.username }}</p>
            <p class="bio" v-if="form.bio">{{ form.bio }}</p>
            <p class="bio" v-else>暂无个人简介</p>
          </div>
          <el-divider />
          <ul class="profile-stats">
            <li>
              <span class="label">账号创建时间</span>
              <span class="value">{{ createdAt }}</span>
            </li>
            <li>
              <span class="label">最近更新</span>
              <span class="value">{{ updatedAt }}</span>
            </li>
            <li>
              <span class="label">联系邮箱</span>
              <span class="value">{{ form.email || '未填写' }}</span>
            </li>
          </ul>
        </el-card>
      </el-col>
      <el-col :span="16" :xs="24">
        <el-card shadow="never" class="profile-card">
          <template #header>
            <div class="card-header">
              <span>个人资料</span>
              <el-tag type="success" v-if="saved">已保存</el-tag>
            </div>
          </template>
          <el-form
            ref="profileFormRef"
            :model="form"
            :rules="rules"
            label-position="top"
            class="profile-form"
          >
            <el-row :gutter="16">
              <el-col :span="12" :xs="24">
                <el-form-item label="用户名" prop="username">
                  <el-input v-model="form.username" disabled />
                </el-form-item>
              </el-col>
              <el-col :span="12" :xs="24">
                <el-form-item label="姓名" prop="full_name">
                  <el-input v-model="form.full_name" placeholder="请输入姓名" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="16">
              <el-col :span="12" :xs="24">
                <el-form-item label="邮箱" prop="email">
                  <el-input v-model="form.email" placeholder="请输入邮箱" />
                </el-form-item>
              </el-col>
              <el-col :span="12" :xs="24">
                <el-form-item label="头像地址" prop="avatar_url">
                  <el-input v-model="form.avatar_url" placeholder="请输入头像图片URL" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-form-item label="个人简介" prop="bio">
              <el-input
                v-model="form.bio"
                type="textarea"
                :rows="5"
                maxlength="300"
                show-word-limit
                placeholder="请简单介绍自己，便于团队成员了解您"
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="authLoading" @click="handleSubmit">
                保存修改
              </el-button>
              <el-button @click="resetForm">重置</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import dayjs from 'dayjs'
import { ElMessage } from 'element-plus'
import { storeToRefs } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import { UserFilled } from '@element-plus/icons-vue'

const authStore = useAuthStore()
const { user, loading } = storeToRefs(authStore)

const profileFormRef = ref(null)
const form = reactive({
  username: '',
  email: '',
  full_name: '',
  bio: '',
  avatar_url: '',
  created_at: '',
  updated_at: ''
})

const rules = {
  email: [
    { type: 'email', message: '邮箱格式不正确', trigger: 'blur' }
  ],
  bio: [
    { max: 300, message: '简介不超过300字', trigger: 'blur' }
  ],
  avatar_url: [
    {
      validator: (_, value, callback) => {
        if (!value) {
          callback()
          return
        }
        try {
          // eslint-disable-next-line no-new
          new URL(value)
          callback()
        } catch (error) {
          callback(new Error('请输入合法的URL地址'))
        }
      },
      trigger: 'blur'
    }
  ]
}

const authLoading = computed(() => loading.value)
const saved = ref(false)

const displayName = computed(() => form.full_name || form.username || '用户')
const createdAt = computed(() => (form.created_at ? dayjs(form.created_at).format('YYYY-MM-DD HH:mm') : '未知'))
const updatedAt = computed(() => (form.updated_at ? dayjs(form.updated_at).format('YYYY-MM-DD HH:mm') : '未知'))

const syncForm = (profile) => {
  form.username = profile?.username || ''
  form.email = profile?.email || ''
  form.full_name = profile?.full_name || ''
  form.bio = profile?.bio || ''
  form.avatar_url = profile?.avatar_url || ''
  form.created_at = profile?.created_at || ''
  form.updated_at = profile?.updated_at || ''
}

watch(
  () => user.value,
  (value) => {
    if (value) {
      syncForm(value)
    }
  },
  { immediate: true }
)

onMounted(async () => {
  if (!user.value) {
    await authStore.fetchProfile()
  }
})

const handleSubmit = () => {
  profileFormRef.value?.validate(async (valid) => {
    if (!valid) return
    const success = await authStore.updateProfile({
      email: form.email,
      full_name: form.full_name,
      bio: form.bio,
      avatar_url: form.avatar_url
    })
    if (success) {
      saved.value = true
      setTimeout(() => {
        saved.value = false
      }, 2000)
    }
  })
}

const resetForm = () => {
  syncForm(user.value)
  ElMessage.info('已重置为最新保存信息')
}
</script>

<style scoped lang="scss">
.profile-page {
  padding: $spacing-6;
}

.profile-card {
  border-radius: $border-radius-lg;
}

.profile-avatar {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: $spacing-3;

  h2 {
    margin: 0;
    font-size: 24px;
  }

  .username {
    margin: 0;
    color: #909399;
  }

  .bio {
    font-size: 14px;
    color: #606266;
    line-height: 1.6;
  }
}

.profile-stats {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: $spacing-3;

  .label {
    display: block;
    color: #909399;
    font-size: 13px;
    margin-bottom: 4px;
  }

  .value {
    font-size: 15px;
    color: #303133;
    font-weight: 500;
  }
}

.profile-form {
  margin-top: $spacing-4;
}

@media (max-width: $breakpoint-md) {
  .profile-page {
    padding: $spacing-4;
  }
}
</style>

