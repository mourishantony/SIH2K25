import axios from 'axios';

const API_URL = '/api';


const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});


api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);


api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  login: (username, password) => 
    api.post('/auth/login', { username, password }),
  register: (username, email, password) => 
    api.post('/auth/register', { username, email, password }),
  getMe: () => 
    api.get('/auth/me'),
  changePassword: (oldPassword, newPassword) => 
    api.post('/auth/change-password', null, { 
      params: { old_password: oldPassword, new_password: newPassword } 
    }),
};


export const dashboardAPI = {
  getStats: () => 
    api.get('/dashboard/stats'),
  getRecentActivity: (limit = 10) => 
    api.get('/dashboard/recent-activity', { params: { limit } }),
  getMDRSummary: () => 
    api.get('/dashboard/mdr-summary'),
  getContactTrends: (days = 7) => 
    api.get('/dashboard/contact-trends', { params: { days } }),
};


export const personsAPI = {
  getAll: (params = {}) => 
    api.get('/persons/', { params }),
  getById: (id) => 
    api.get(`/persons/${id}`),
  getByName: (name) => 
    api.get(`/persons/by-name/${name}`),
  create: (data) => 
    api.post('/persons/', data),
  update: (id, data) => 
    api.put(`/persons/${id}`, data),
  delete: (id) => 
    api.delete(`/persons/${id}`),
  getContacts: (id, limit = 50) => 
    api.get(`/persons/${id}/contacts`, { params: { limit } }),
  getRoles: () => 
    api.get('/persons/roles'),
};


export const faceAPI = {
  getSettings: () => 
    api.get('/face/settings'),
  uploadImages: (personName, images, autoTrain = true) => 
    api.post('/face/upload', { person_name: personName, images, auto_train: autoTrain }),
  train: (personName, useGpu = null) => 
    api.post('/face/train', { person_name: personName, use_gpu: useGpu }),
  retrain: (personName, useGpu = null) => 
    api.post('/face/retrain', { person_name: personName, use_gpu: useGpu }),
  getTrainingStatus: (personName) => 
    api.get(`/face/status/${encodeURIComponent(personName)}`),
  getStatus: (personName) => 
    api.get(`/face/status/${encodeURIComponent(personName)}`),
  deleteImages: (personName) => 
    api.delete(`/face/images/${personName}`),
  getRegistered: () => 
    api.get('/face/registered'),
};


export const mdrAPI = {
  getPatients: () => 
    api.get('/mdr/patients'),
  getPatient: (name) => 
    api.get(`/mdr/patients/${name}`),
  getPathogens: () =>
    api.get('/mdr/pathogens'),
  markAsMDR: (personName, pathogenType = 'Other', notes = '') => 
    api.post('/mdr/mark', { person_name: personName, pathogen_type: pathogenType, notes }),
  unmarkMDR: (personName) => 
    api.post('/mdr/unmark', { person_name: personName }),
  updatePatient: (name, data) => 
    api.put(`/mdr/patients/${name}`, data),
  checkStatus: (name) => 
    api.get(`/mdr/check/${name}`),
  getEligible: () => 
    api.get('/mdr/eligible'),
  getContacts: (name, limit = 50) => 
    api.get(`/mdr/contacts/${name}`, { params: { limit } }),
};


export const alertsAPI = {
  getAll: (limit = 100, unreadOnly = false) => 
    api.get('/alerts/', { params: { limit, unread_only: unreadOnly } }),
  getUnread: () => 
    api.get('/alerts/unread'),
  getCounts: () => 
    api.get('/alerts/count'),
  getDetail: (id) => 
    api.get(`/alerts/${id}`),
  markAsRead: (id) => 
    api.post(`/alerts/${id}/read`),
  markAllAsRead: () => 
    api.post('/alerts/read-all'),
  delete: (id) => 
    api.delete(`/alerts/${id}`),
  deleteAll: () => 
    api.delete('/alerts/all'),
  getForPatient: (patientName) => 
    api.get(`/alerts/patient/${patientName}`),
  getCollisionAlerts: (limit = 50) => 
    api.get('/alerts/collision/recent', { params: { limit } }),
};


export const unknownAPI = {
  getAll: (params = {}) => 
    api.get('/unknown/', { params }),
  getById: (tempId) => 
    api.get(`/unknown/${tempId}`),
  getContacts: (tempId, limit = 50) => 
    api.get(`/unknown/${tempId}/contacts`, { params: { limit } }),
  markAsKnown: (tempId, personName) => 
    api.post(`/unknown/${tempId}/mark-known`, { person_name: personName }),
  delete: (tempId) => 
    api.delete(`/unknown/${tempId}`),
  register: (tempId, data) => 
    api.post(`/unknown/${tempId}/register`, data),
  getSettings: () => 
    api.get('/unknown/settings'),
};


export const monitoringAPI = {
 
  uploadVideo: (file, videoType = 'front', onUploadProgress) => {
    const formData = new FormData();
    formData.append('video_file', file);
    formData.append('video_type', videoType);
    return api.post('/monitoring/upload-video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress,
    });
  },
  listVideos: () => 
    api.get('/monitoring/uploaded-videos'),
  deleteVideo: (filename) => 
    api.delete(`/monitoring/uploaded-videos/${filename}`),
  
  
  getStatus: () =>
    api.get('/monitoring/status'),
  startMonitoring: (config) => 
    api.post('/monitoring/start', config),
  stopMonitoring: () => 
    api.post('/monitoring/stop'),
  getConfig: () =>
    api.get('/monitoring/config'),
  
 
  getWebSocketUrl: () => {
    
    
    const isDev = import.meta.env.DEV;
    if (isDev) {
      return 'ws://localhost:8000/api/monitoring/ws';
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/api/monitoring/ws`;
  },
};


export const usersAPI = {
  getAll: () => 
    api.get('/auth/users'),
  create: (userData) => 
    api.post('/auth/users', userData),
  update: (userId, userData) => 
    api.put(`/auth/users/${userId}`, userData),
  delete: (userId) => 
    api.delete(`/auth/users/${userId}`),
  getRoles: () => 
    api.get('/auth/roles'),
};


export const pathogensAPI = {
  getAll: () => 
    api.get('/pathogens/'),
  getByName: (name) => 
    api.get(`/pathogens/${name}`),
  create: (data) => 
    api.post('/pathogens/', data),
  update: (name, data) => 
    api.put(`/pathogens/${name}`, data),
  delete: (name) => 
    api.delete(`/pathogens/${name}`),
};

export default api;
