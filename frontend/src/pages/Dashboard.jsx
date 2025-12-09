import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { dashboardAPI, alertsAPI } from '../api';
import { useAuth } from '../context/AuthContext';
import { 
  Users, 
  UserCheck, 
  Stethoscope, 
  HardHat, 
  Eye,
  Heart,
  AlertTriangle,
  Activity,
  TrendingUp,
  Bell,
  ArrowRight,
  BarChart3,
  PieChart as PieChartIcon
} from 'lucide-react';
import { format } from 'date-fns';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

export default function Dashboard() {
  const { hasPermission } = useAuth();
  const [stats, setStats] = useState(null);
  const [recentActivity, setRecentActivity] = useState([]);
  const [mdrSummary, setMDRSummary] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [statsRes, activityRes, mdrRes] = await Promise.all([
        dashboardAPI.getStats(),
        dashboardAPI.getRecentActivity(10),
        dashboardAPI.getMDRSummary()
      ]);
      
      setStats(statsRes.data);
      setRecentActivity(activityRes.data);
      setMDRSummary(mdrRes.data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  const statCards = [
    {
      label: 'Total Patients',
      value: stats?.persons?.patients || 0,
      icon: Users,
      color: 'primary',
      bgColor: 'bg-primary-50',
      textColor: 'text-primary-600'
    },
    {
      label: 'Doctors',
      value: stats?.persons?.doctors || 0,
      icon: Stethoscope,
      color: 'success',
      bgColor: 'bg-success-50',
      textColor: 'text-success-600'
    },
    {
      label: 'Visitors',
      value: stats?.persons?.visitors || 0,
      icon: Eye,
      color: 'warning',
      bgColor: 'bg-warning-50',
      textColor: 'text-warning-600'
    },
    {
      label: 'Nurses',
      value: stats?.persons?.nurses || 0,
      icon: Heart,
      color: 'pink',
      bgColor: 'bg-pink-50',
      textColor: 'text-pink-600'
    },
    {
      label: 'Workers',
      value: stats?.persons?.workers || 0,
      icon: HardHat,
      color: 'gray',
      bgColor: 'bg-gray-100',
      textColor: 'text-gray-600'
    },
    {
      label: 'MDR Patients',
      value: stats?.mdr?.total || 0,
      icon: AlertTriangle,
      color: 'danger',
      bgColor: 'bg-danger-50',
      textColor: 'text-danger-600'
    },
    {
      label: 'Unread Alerts',
      value: stats?.alerts?.unread || 0,
      icon: Bell,
      color: 'danger',
      bgColor: 'bg-danger-50',
      textColor: 'text-danger-600'
    }
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Dashboard</h1>
          <p className="text-gray-500">Overview of the contact tracing system</p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="card">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <Icon className={`h-6 w-6 ${stat.textColor}`} />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-800">{stat.value}</p>
                  <p className="text-sm text-gray-500">{stat.label}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Registered Persons by Role - Bar Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">Registered Persons by Role</h2>
            <BarChart3 className="h-5 w-5 text-gray-400" />
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[
                  { name: 'Patients', count: stats?.persons?.patients || 0, fill: '#3b82f6' },
                  { name: 'Doctors', count: stats?.persons?.doctors || 0, fill: '#22c55e' },
                  { name: 'Visitors', count: stats?.persons?.visitors || 0, fill: '#f59e0b' },
                  { name: 'Nurses', count: stats?.persons?.nurses || 0, fill: '#ec4899' },
                  { name: 'Workers', count: stats?.persons?.workers || 0, fill: '#6b7280' },
                ]}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="name" 
                  tick={{ fill: '#6b7280', fontSize: 12 }}
                  axisLine={{ stroke: '#e5e7eb' }}
                />
                <YAxis 
                  tick={{ fill: '#6b7280', fontSize: 12 }}
                  axisLine={{ stroke: '#e5e7eb' }}
                  allowDecimals={false}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                  formatter={(value) => [value, 'Count']}
                />
                <Bar 
                  dataKey="count" 
                  radius={[4, 4, 0, 0]}
                >
                  {[
                    { name: 'Patients', fill: '#3b82f6' },
                    { name: 'Doctors', fill: '#22c55e' },
                    { name: 'Visitors', fill: '#f59e0b' },
                    { name: 'Nurses', fill: '#ec4899' },
                    { name: 'Workers', fill: '#6b7280' },
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid grid-cols-5 gap-2 text-center">
            <div className="p-2 bg-blue-50 rounded-lg">
              <p className="text-lg font-bold text-blue-600">{stats?.persons?.patients || 0}</p>
              <p className="text-xs text-gray-500">Patients</p>
            </div>
            <div className="p-2 bg-green-50 rounded-lg">
              <p className="text-lg font-bold text-green-600">{stats?.persons?.doctors || 0}</p>
              <p className="text-xs text-gray-500">Doctors</p>
            </div>
            <div className="p-2 bg-amber-50 rounded-lg">
              <p className="text-lg font-bold text-amber-600">{stats?.persons?.visitors || 0}</p>
              <p className="text-xs text-gray-500">Visitors</p>
            </div>
            <div className="p-2 bg-pink-50 rounded-lg">
              <p className="text-lg font-bold text-pink-600">{stats?.persons?.nurses || 0}</p>
              <p className="text-xs text-gray-500">Nurses</p>
            </div>
            <div className="p-2 bg-gray-100 rounded-lg">
              <p className="text-lg font-bold text-gray-600">{stats?.persons?.workers || 0}</p>
              <p className="text-xs text-gray-500">Workers</p>
            </div>
          </div>
        </div>

        {/* MDR vs Non-MDR - Pie Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">MDR Patient Distribution</h2>
            <PieChartIcon className="h-5 w-5 text-gray-400" />
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'MDR Patients', value: stats?.mdr?.total || 0, color: '#ef4444' },
                    { name: 'Non-MDR', value: Math.max((stats?.persons?.total || 0) - (stats?.mdr?.total || 0), 0), color: '#22c55e' },
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => percent > 0 ? `${(percent * 100).toFixed(0)}%` : ''}
                  labelLine={false}
                >
                  <Cell fill="#ef4444" />
                  <Cell fill="#22c55e" />
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                  formatter={(value, name) => [value, name]}
                />
                <Legend 
                  verticalAlign="bottom" 
                  height={36}
                  formatter={(value, entry) => (
                    <span style={{ color: '#374151', fontSize: '12px' }}>{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="p-3 bg-red-50 rounded-lg border border-red-100">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-red-500" />
                <div>
                  <p className="text-2xl font-bold text-red-600">{stats?.mdr?.total || 0}</p>
                  <p className="text-sm text-gray-500">MDR Patients</p>
                </div>
              </div>
            </div>
            <div className="p-3 bg-green-50 rounded-lg border border-green-100">
              <div className="flex items-center gap-2">
                <UserCheck className="h-5 w-5 text-green-500" />
                <div>
                  <p className="text-2xl font-bold text-green-600">
                    {Math.max((stats?.persons?.total || 0) - (stats?.mdr?.total || 0), 0)}
                  </p>
                  <p className="text-sm text-gray-500">Non-MDR Persons</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">Recent Activity</h2>
            <Activity className="h-5 w-5 text-gray-400" />
          </div>
          
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {recentActivity.length === 0 ? (
              <p className="text-center text-gray-500 py-4">No recent activity</p>
            ) : (
              recentActivity.map((activity, index) => (
                <div key={index} className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50">
                  <div className={`p-2 rounded-full flex-shrink-0 ${
                    activity.type === 'alert' ? 'bg-danger-100' :
                    activity.type === 'contact' ? 'bg-warning-100' :
                    'bg-success-100'
                  }`}>
                    {activity.type === 'alert' ? (
                      <AlertTriangle className="h-4 w-4 text-danger-600" />
                    ) : activity.type === 'contact' ? (
                      <Users className="h-4 w-4 text-warning-600" />
                    ) : (
                      <UserCheck className="h-4 w-4 text-success-600" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-800 truncate">{activity.description}</p>
                    <p className="text-xs text-gray-500">
                      {format(new Date(activity.timestamp), 'MMM d, HH:mm')}
                      {activity.risk_percent > 0 && (
                        <span className={`ml-2 ${
                          activity.risk_percent >= 70 ? 'text-danger-600' :
                          activity.risk_percent >= 40 ? 'text-warning-600' :
                          'text-success-600'
                        }`}>
                          • Risk: {activity.risk_percent.toFixed(1)}%
                        </span>
                      )}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* MDR Patients Summary */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">MDR Patients</h2>
            <Link to="/mdr" className="text-primary-600 hover:underline text-sm flex items-center gap-1">
              View all <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
          
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {mdrSummary.length === 0 ? (
              <div className="text-center py-8">
                <AlertTriangle className="h-12 w-12 text-gray-300 mx-auto mb-2" />
                <p className="text-gray-500">No MDR patients marked</p>
                <Link to="/mdr" className="text-primary-600 hover:underline text-sm mt-2 inline-block">
                  Mark a patient as MDR
                </Link>
              </div>
            ) : (
              mdrSummary.map((patient, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-danger-50 border border-danger-100">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-full bg-danger-100">
                      <AlertTriangle className="h-4 w-4 text-danger-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-800">{patient.name}</p>
                      <p className="text-xs text-gray-500">
                        {patient.alert_count} alert{patient.alert_count !== 1 ? 's' : ''}
                      </p>
                    </div>
                  </div>
                  {patient.latest_alert && (
                    <div className="text-right">
                      <p className="text-xs text-gray-500">Last contact</p>
                      <p className="text-sm text-gray-700">{patient.latest_alert.contacted_person}</p>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {hasPermission('register_person') && (
          <Link to="/register" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center">
            <UserCheck className="h-8 w-8 text-primary-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">Register Person</p>
            <p className="text-sm text-gray-500">Add new patient/doctor</p>
          </Link>
          )}
          
          {hasPermission('registered_persons') && (
          <Link to="/persons" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center">
            <Users className="h-8 w-8 text-primary-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">View Persons</p>
            <p className="text-sm text-gray-500">Manage registered persons</p>
          </Link>
          )}
          
          {hasPermission('mdr_management') && (
          <Link to="/mdr" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-danger-300 hover:bg-danger-50 transition-colors text-center">
            <AlertTriangle className="h-8 w-8 text-danger-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">MDR Management</p>
            <p className="text-sm text-gray-500">Mark MDR patients</p>
          </Link>
          )}
          
          {hasPermission('alerts') && (
          <Link to="/alerts" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-warning-300 hover:bg-warning-50 transition-colors text-center relative">
            <Bell className="h-8 w-8 text-warning-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">View Alerts</p>
            <p className="text-sm text-gray-500">Check contact alerts</p>
            {stats?.alerts?.unread > 0 && (
              <span className="absolute top-2 right-2 bg-danger-500 text-white text-xs px-2 py-0.5 rounded-full">
                {stats.alerts.unread} new
              </span>
            )}
          </Link>
          )}
        </div>
      </div>
    </div>
  );
}
