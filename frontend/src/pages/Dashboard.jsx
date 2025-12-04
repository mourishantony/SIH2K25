import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { dashboardAPI, alertsAPI } from '../api';
import { 
  Users, 
  UserCheck, 
  Stethoscope, 
  HardHat, 
  Eye,
  AlertTriangle,
  Activity,
  TrendingUp,
  Bell,
  ArrowRight
} from 'lucide-react';
import { format } from 'date-fns';

export default function Dashboard() {
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
          <Link to="/register-person" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center">
            <UserCheck className="h-8 w-8 text-primary-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">Register Person</p>
            <p className="text-sm text-gray-500">Add new patient/doctor</p>
          </Link>
          
          <Link to="/persons" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center">
            <Users className="h-8 w-8 text-primary-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">View Persons</p>
            <p className="text-sm text-gray-500">Manage registered persons</p>
          </Link>
          
          <Link to="/mdr" className="p-4 rounded-lg border-2 border-dashed border-gray-200 hover:border-danger-300 hover:bg-danger-50 transition-colors text-center">
            <AlertTriangle className="h-8 w-8 text-danger-600 mx-auto mb-2" />
            <p className="font-medium text-gray-800">MDR Management</p>
            <p className="text-sm text-gray-500">Mark MDR patients</p>
          </Link>
          
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
        </div>
      </div>
    </div>
  );
}
