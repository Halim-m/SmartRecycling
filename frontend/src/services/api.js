import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const classifyImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/classify', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getCategories = async () => {
  const response = await api.get('/categories');
  return response.data;
};

export const getModelInfo = async () => {
  const response = await api.get('/model/info');
  return response.data;
};

export default api;
