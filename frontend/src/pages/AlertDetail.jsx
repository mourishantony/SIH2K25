import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { alertsAPI } from '../api';
import { 
  AlertTriangle, ArrowLeft, User, Calendar, 
  Clock, Mail, Check, Image, Ruler
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';

export default function AlertDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlert();
  }, [id]);

  const fetchAlert = async () => {
    try {
      setLoading(true);
      
      const res = await alertsAPI.getAll();
      const alerts = res.data.alerts || res.data || [];
      const foundAlert = alerts.find(a => (a.id === id || a._id === id));
      if (foundAlert) {
        setAlert(foundAlert);
      } else {
        toast.error('Alert not found');
        navigate('/alerts');
      }
    } catch (error) {
      toast.error('Failed to fetch alert');
      navigate('/alerts');
    } finally {
      setLoading(false);
    }
  };

  const handleMarkAsRead = async () => {
    try {
      await alertsAPI.markAsRead(alert.id || alert._id);
      toast.success('Alert marked as read');
      setAlert({ ...alert, is_read: true });
    } catch (error) {
      toast.error('Failed to mark as read');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (!alert) {
    return (
      <div className="text-center py-20">
        <AlertTriangle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-700">Alert not found</h2>
        <Link to="/alerts" className="text-primary-600 hover:underline mt-2 block">
          Back to alerts
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fadeIn">
      <div className="flex items-center gap-4">
        <button 
          onClick={() => navigate('/alerts')}
          className="p-2 hover:bg-gray-100 rounded-lg"
        >
          <ArrowLeft className="h-5 w-5" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Alert Details</h1>
          <p className="text-gray-500">
            {alert.timestamp && format(new Date(alert.timestamp), 'MMMM d, yyyy HH:mm:ss')}
          </p>
        </div>
      </div>

      <div className="card">
        {/* Status Badge */}
        <div className="flex items-center justify-between mb-6">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            alert.is_read 
              ? 'bg-gray-100 text-gray-700' 
              : 'bg-danger-100 text-danger-700'
          }`}>
            <AlertTriangle className="h-5 w-5" />
            <span className="font-medium">
              {alert.is_read ? 'Read' : 'Unread'} Alert
            </span>
          </div>
          
          {!alert.is_read && (
            <button onClick={handleMarkAsRead} className="btn-primary flex items-center gap-2">
              <Check className="h-4 w-4" />
              Mark as Read
            </button>
          )}
        </div>

        {/* Persons Involved */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="p-6 bg-red-50 rounded-xl border border-red-100">
            <p className="text-xs text-red-600 uppercase font-medium tracking-wide mb-3">
              MDR Patient
            </p>
            <div className="flex items-center gap-4">
              <div className="h-14 w-14 rounded-full bg-red-100 flex items-center justify-center">
                <User className="h-7 w-7 text-red-600" />
              </div>
              <div>
                <span className="text-xl font-semibold text-gray-900">{alert.mdr_patient}</span>
                <p className="text-sm text-red-600">Diagnosed MDR</p>
              </div>
            </div>
          </div>
          
          <div className="p-6 bg-blue-50 rounded-xl border border-blue-100">
            <p className="text-xs text-blue-600 uppercase font-medium tracking-wide mb-3">
              Contact Person
            </p>
            <div className="flex items-center gap-4">
              <div className="h-14 w-14 rounded-full bg-blue-100 flex items-center justify-center">
                <User className="h-7 w-7 text-blue-600" />
              </div>
              <div>
                <span className="text-xl font-semibold text-gray-900">{alert.contacted_person || alert.contact_name || 'Unknown'}</span>
                <p className="text-sm text-blue-600">At Risk</p>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Assessment */}
        <div className="mb-6 p-6 rounded-xl border-2 border-dashed ${
          (alert.risk_percent || 0) >= 40 
            ? 'border-red-300 bg-red-50' 
            : (alert.risk_percent || 0) >= 20 
              ? 'border-yellow-300 bg-yellow-50'
              : 'border-green-300 bg-green-50'
        }">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 mb-1">Risk Assessment</p>
              <p className={`text-3xl font-bold ${
                (alert.risk_percent || 0) >= 40 
                  ? 'text-red-600' 
                  : (alert.risk_percent || 0) >= 20 
                    ? 'text-yellow-600'
                    : 'text-green-600'
              }`}>
                {(alert.risk_percent || 0).toFixed(1)}%
              </p>
            </div>
            <div className={`px-4 py-2 rounded-full text-sm font-semibold ${
              (alert.risk_percent || 0) >= 40 
                ? 'bg-red-100 text-red-800' 
                : (alert.risk_percent || 0) >= 20 
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-green-100 text-green-800'
            }`}>
              {(alert.risk_percent || 0) >= 40 ? 'High Risk' : (alert.risk_percent || 0) >= 20 ? 'Medium Risk' : 'Low Risk'}
            </div>
          </div>
          {alert.contact_count > 1 && (
            <p className="mt-2 text-sm text-gray-600">
              Based on {alert.contact_count} recorded interactions
            </p>
          )}
        </div>

        {/* Contact Details */}
        <div className="space-y-4 mb-6">
          <h3 className="text-lg font-semibold text-gray-800">Contact Details</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                <Calendar className="h-4 w-4" />
                Date
              </div>
              <p className="font-medium">
                {alert.timestamp 
                  ? format(new Date(alert.timestamp), 'MMM d, yyyy')
                  : 'N/A'
                }
              </p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                <Clock className="h-4 w-4" />
                Time
              </div>
              <p className="font-medium">
                {alert.timestamp 
                  ? format(new Date(alert.timestamp), 'HH:mm:ss')
                  : 'N/A'
                }
              </p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                <Clock className="h-4 w-4" />
                Duration
              </div>
              <p className="font-medium">
                {alert.duration_seconds 
                  ? `${Math.round(alert.duration_seconds)} seconds` 
                  : 'N/A'
                }
              </p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                <Ruler className="h-4 w-4" />
                Distance
              </div>
              <p className="font-medium">
                {alert.distance_meters !== undefined && alert.distance_meters !== null
                  ? `${alert.distance_meters.toFixed(2)} m`
                  : alert.min_distance_meters !== undefined && alert.min_distance_meters !== null
                    ? `${alert.min_distance_meters.toFixed(2)} m (min)`
                    : 'N/A'
                }
              </p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                <AlertTriangle className="h-4 w-4" />
                Interactions
              </div>
              <p className="font-medium">
                {alert.contact_count ? `${alert.contact_count} contacts` : '1 contact'}
              </p>
            </div>
          </div>
        </div>

        {/* Email Status */}
        <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg mb-6">
          <Mail className={`h-5 w-5 ${alert.email_sent ? 'text-green-600' : 'text-gray-400'}`} />
          <div>
            <p className="font-medium text-gray-800">
              Email Notification
            </p>
            <p className="text-sm text-gray-500">
              {alert.email_sent 
                ? 'Email notification was sent successfully' 
                : 'No email notification was sent'
              }
            </p>
          </div>
        </div>

        {/* Snapshot */}
        {alert.snapshot_base64 && (
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <Image className="h-5 w-5" />
              Incident Snapshot
            </h3>
            <img
              src={alert.snapshot_base64.startsWith('data:') 
                ? alert.snapshot_base64 
                : `data:image/jpeg;base64,${alert.snapshot_base64}`
              }
              alt="Contact incident snapshot"
              className="w-full rounded-lg border border-gray-200 shadow-sm"
            />
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <Link to="/alerts" className="btn-secondary">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Alerts
        </Link>
        <Link to="/mdr" className="btn-secondary">
          View MDR Patients
        </Link>
      </div>
    </div>
  );
}
