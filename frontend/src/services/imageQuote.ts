import axios from 'axios';

const API_BASE = 'http://localhost:8004';

export async function processImageForQuote(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${API_BASE}/api/image/process-image`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}
