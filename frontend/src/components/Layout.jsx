import { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { alertsAPI } from '../api';
import { 
  LayoutDashboard, 
  UserPlus, 
  Users, 
  AlertTriangle, 
  Bell, 
  LogOut,
  Menu,
  X,
  Activity,
  UserX,
  Camera,
  ShieldAlert,
  Settings
} from 'lucide-react';
import toast from 'react-hot-toast';


const ICON_MAP = {
  LayoutDashboard,
  UserPlus,
  Users,
  Camera,
  UserX,
  ShieldAlert,
  Bell,
  Settings,
};

export default function Layout() {
  const { user, logout, allowedNavItems, hasPermission } = useAuth();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [unreadAlerts, setUnreadAlerts] = useState(0);
  const [showAlertPopup, setShowAlertPopup] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState([]);


  useEffect(() => {
   
    if (!hasPermission('alerts')) {
      setUnreadAlerts(0);
      setRecentAlerts([]);
      return;
    }

    const fetchUnreadAlerts = async () => {
      try {
        const response = await alertsAPI.getUnread();
        const alerts = response.data;
        
        
        if (Array.isArray(alerts)) {
          setUnreadAlerts(alerts.length);
          setRecentAlerts(alerts.slice(0, 5));
          
         
          if (alerts.length > 0) {
            const latestAlert = alerts[0];
            const alertTime = new Date(latestAlert.timestamp || latestAlert.created_at);
            const now = new Date();
            const diffSeconds = (now - alertTime) / 1000;
            
            if (diffSeconds < 30) {
              toast.custom((t) => (
                <div className={`${t.visible ? 'animate-fadeIn' : 'opacity-0'} bg-danger-600 text-white p-4 rounded-lg shadow-lg max-w-md`}>
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-6 w-6 flex-shrink-0" />
                    <div>
                      <p className="font-bold">🚨 MDR Contact Alert!</p>
                      <p className="text-sm mt-1">
                        {latestAlert.mdr_patient} contacted {latestAlert.contact_name}
                      </p>
                    </div>
                  </div>
                </div>
              ), { duration: 6000 });
            }
          }
        } else {
          setUnreadAlerts(alerts.unread_count || 0);
          setRecentAlerts((alerts.alerts || []).slice(0, 5));
        }
      } catch (error) {
        console.error('Error fetching alerts:', error);
      }
    };

    fetchUnreadAlerts();
    
    
    const interval = setInterval(fetchUnreadAlerts, 30000);
    return () => clearInterval(interval);
  }, [hasPermission]);

  const handleLogout = () => {
    logout();
    toast.success('Logged out successfully');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed top-0 left-0 z-30 h-full w-64 bg-white shadow-lg transform transition-transform duration-200
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0
      `}>
        <div className="p-4 border-b">
          <div className="flex items-center gap-3">
            <Activity className="h-8 w-8 text-primary-600" />
            <div>
              <h1 className="font-bold text-lg text-gray-800">Contact Tracing</h1>
              <p className="text-xs text-gray-500">SIH 2025</p>
            </div>
          </div>
        </div>

        <nav className="p-4 space-y-1">
          {allowedNavItems.map((item) => {
            const Icon = ICON_MAP[item.icon] || LayoutDashboard;
            const isActive = location.pathname === item.path;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`
                  flex items-center gap-3 px-4 py-3 rounded-lg transition-colors relative
                  ${isActive 
                    ? 'bg-primary-50 text-primary-700 font-medium' 
                    : 'text-gray-600 hover:bg-gray-50'
                  }
                `}
              >
                <Icon className="h-5 w-5" />
                {item.label}
                {item.path === '/alerts' && unreadAlerts > 0 && (
                  <span className="ml-auto bg-danger-500 text-white text-xs px-2 py-0.5 rounded-full">
                    {unreadAlerts}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t">
          <div className="flex items-center gap-3 mb-4">
            <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
              <span className="text-primary-700 font-medium">
                {user?.username?.charAt(0).toUpperCase()}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-gray-800 truncate">{user?.username}</p>
              <p className="text-xs text-gray-500 truncate">{user?.email}</p>
              <p className="text-xs text-primary-600 capitalize">{user?.role?.replace('_', ' ')}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <LogOut className="h-4 w-4" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="lg:ml-64">
        {/* Top bar */}
        <header className="bg-white shadow-sm sticky top-0 z-10">
          <div className="flex items-center justify-between px-4 py-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100"
            >
              <Menu className="h-6 w-6" />
            </button>

            <div className="flex-1 lg:flex-none" />

            <div className="flex items-center gap-4">
              {/* Alert bell */}
              <div className="relative">
                <button
                  onClick={() => setShowAlertPopup(!showAlertPopup)}
                  className="p-2 rounded-lg hover:bg-gray-100 relative"
                >
                  <Bell className="h-6 w-6 text-gray-600" />
                  {unreadAlerts > 0 && (
                    <span className="notification-badge">{unreadAlerts}</span>
                  )}
                </button>

                {/* Alert popup */}
                {showAlertPopup && (
                  <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border z-50 animate-fadeIn">
                    <div className="p-3 border-b flex items-center justify-between">
                      <span className="font-medium">Recent Alerts</span>
                      <Link 
                        to="/alerts" 
                        className="text-sm text-primary-600 hover:underline"
                        onClick={() => setShowAlertPopup(false)}
                      >
                        View all
                      </Link>
                    </div>
                    <div className="max-h-80 overflow-y-auto">
                      {recentAlerts.length === 0 ? (
                        <p className="p-4 text-center text-gray-500 text-sm">
                          No unread alerts
                        </p>
                      ) : (
                        recentAlerts.map((alert, idx) => (
                          <Link
                            key={alert.id || alert._id || idx}
                            to={`/alerts/${alert.id || alert._id}`}
                            onClick={() => setShowAlertPopup(false)}
                            className="block p-3 hover:bg-gray-50 border-b last:border-b-0"
                          >
                            <div className="flex items-start gap-2">
                              <AlertTriangle className="h-4 w-4 text-danger-500 mt-0.5 flex-shrink-0" />
                              <div className="min-w-0">
                                <p className="text-sm font-medium text-gray-800 truncate">
                                  {alert.mdr_patient} → {alert.contact_name}
                                </p>
                                <p className="text-xs text-gray-500">
                                  {alert.contact_type || 'Contact'} • {alert.duration_seconds ? `${alert.duration_seconds}s` : 'N/A'}
                                  {(alert.min_distance_meters !== undefined && alert.min_distance_meters !== null) 
                                    ? ` • ${alert.min_distance_meters.toFixed(2)}m`
                                    : (alert.distance_meters !== undefined && alert.distance_meters !== null)
                                      ? ` • ${alert.distance_meters.toFixed(2)}m`
                                      : ''}
                                </p>
                              </div>
                            </div>
                          </Link>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
