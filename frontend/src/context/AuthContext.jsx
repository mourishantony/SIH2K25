import { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { authAPI } from '../api';

const AuthContext = createContext(null);


const ROLE_PERMISSIONS = {
  admin: [
    'dashboard', 'registered_persons', 'unknown_persons', 
    'alerts', 'user_management', 'monitoring'
  ],
  ehr_user: [
    'dashboard', 'registered_persons', 'mdr_management', 'alerts'
  ],
  officer: [
    'dashboard', 'register_person', 'registered_persons', 'unknown_persons'
  ],
};


export const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard', permission: 'dashboard', icon: 'LayoutDashboard' },
  { path: '/register', label: 'Register Person', permission: 'register_person', icon: 'UserPlus' },
  { path: '/persons', label: 'Registered Persons', permission: 'registered_persons', icon: 'Users' },
  { path: '/monitoring', label: 'AI Monitoring', permission: 'monitoring', icon: 'Camera' },
  { path: '/unknown', label: 'Unknown Persons', permission: 'unknown_persons', icon: 'UserX' },
  { path: '/mdr', label: 'MDR Management', permission: 'mdr_management', icon: 'ShieldAlert' },
  { path: '/alerts', label: 'Alerts', permission: 'alerts', icon: 'Bell' },
  { path: '/users', label: 'User Management', permission: 'user_management', icon: 'Settings' },
];

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      if (token) {
        try {
          const response = await authAPI.getMe();
          setUser(response.data);
        } catch (error) {
          console.error('Auth init error:', error);
          logout();
        }
      }
      setLoading(false);
    };

    initAuth();
  }, [token]);

  const login = async (username, password) => {
    const response = await authAPI.login(username, password);
    const { access_token, user: userData } = response.data;
    
    localStorage.setItem('token', access_token);
    localStorage.setItem('user', JSON.stringify(userData));
    
    setToken(access_token);
    setUser(userData);
    
    return userData;
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
  };

 
  const permissions = useMemo(() => {
    if (!user?.role) return [];
    return ROLE_PERMISSIONS[user.role] || [];
  }, [user]);

 
  const hasPermission = (permission) => {
    return permissions.includes(permission);
  };


  const isAdmin = useMemo(() => {
    return user?.role === 'admin';
  }, [user]);

 
  const allowedNavItems = useMemo(() => {
    return NAV_ITEMS.filter(item => hasPermission(item.permission));
  }, [permissions]);

  const value = {
    user,
    token,
    loading,
    isAuthenticated: !!token && !!user,
    login,
    logout,
 
    permissions,
    hasPermission,
    isAdmin,
    allowedNavItems,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
