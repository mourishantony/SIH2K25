import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import RegisterPerson from './pages/RegisterPerson';
import PersonsList from './pages/PersonsList';
import MDRManagement from './pages/MDRManagement';
import Alerts from './pages/Alerts';
import AlertDetail from './pages/AlertDetail';
import UnknownPersons from './pages/UnknownPersons';
import MonitoringPage from './pages/MonitoringPage';
import UserManagement from './pages/UserManagement';

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
}

// Role-based route protection
function RoleProtectedRoute({ children, permission }) {
  const { hasPermission, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }
  
  if (!hasPermission(permission)) {
    return <Navigate to="/dashboard" replace />;
  }
  
  return children;
}

function PublicRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }
  
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }
  
  return children;
}

function App() {
  return (
    <Routes>
      {/* Public routes - only login, no self-registration */}
      <Route path="/login" element={
        <PublicRoute>
          <Login />
        </PublicRoute>
      } />
      
      {/* Protected routes */}
      <Route path="/" element={
        <ProtectedRoute>
          <Layout />
        </ProtectedRoute>
      }>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        
        {/* Officer routes */}
        <Route path="register" element={
          <RoleProtectedRoute permission="register_person">
            <RegisterPerson />
          </RoleProtectedRoute>
        } />
        <Route path="monitoring" element={
          <RoleProtectedRoute permission="monitoring">
            <MonitoringPage />
          </RoleProtectedRoute>
        } />
        
        {/* Shared routes */}
        <Route path="persons" element={
          <RoleProtectedRoute permission="registered_persons">
            <PersonsList />
          </RoleProtectedRoute>
        } />
        <Route path="unknown" element={
          <RoleProtectedRoute permission="unknown_persons">
            <UnknownPersons />
          </RoleProtectedRoute>
        } />
        
        {/* EHR user routes */}
        <Route path="mdr" element={
          <RoleProtectedRoute permission="mdr_management">
            <MDRManagement />
          </RoleProtectedRoute>
        } />
        <Route path="alerts" element={
          <RoleProtectedRoute permission="alerts">
            <Alerts />
          </RoleProtectedRoute>
        } />
        <Route path="alerts/:id" element={
          <RoleProtectedRoute permission="alerts">
            <AlertDetail />
          </RoleProtectedRoute>
        } />
        
        {/* Admin only routes */}
        <Route path="users" element={
          <RoleProtectedRoute permission="user_management">
            <UserManagement />
          </RoleProtectedRoute>
        } />
      </Route>
      
      {/* Catch all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
