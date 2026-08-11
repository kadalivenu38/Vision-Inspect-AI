import { useState, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';

export function useAuth() {
  const navigate = useNavigate();
  const [, setRefresh] = useState(0);

  const getUser = useCallback(() => {
    try {
      const raw = localStorage.getItem('user');
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }, []);

  const user = useMemo(() => getUser(), [getUser]);
  const token = localStorage.getItem('token');
  const isAuthenticated = Boolean(token);
  const userName = localStorage.getItem('name') || 'User';
  const userRole = user?.role || '';
  const isQualityEngineer = userRole === 'QA-Engineer';
  const isSupervisor = userRole === 'Supervisor';

  const logout = useCallback(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('name');
    localStorage.removeItem('user');
    navigate('/login');
  }, [navigate]);

  const forceRefresh = useCallback(() => {
    setRefresh((prev) => prev + 1);
  }, []);

  return {
    user,
    token,
    isAuthenticated,
    userName,
    userRole,
    isQualityEngineer,
    isSupervisor,
    logout,
    forceRefresh,
  };
}
