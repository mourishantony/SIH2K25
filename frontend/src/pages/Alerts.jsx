import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { alertsAPI } from '../api';
import { useAuth } from '../context/AuthContext';
import { 
  AlertTriangle, Bell, Check, CheckCheck, Eye, 
  Calendar, User, Users, Filter, RefreshCw, Mail, Image, Trash2, Ruler
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format, formatDistanceToNow } from 'date-fns';

export default function Alerts() {
  const { isAdmin } = useAuth();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, unread, read
  const [selectedAlert, setSelectedAlert] = useState(null);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      const res = filter === 'unread' 
        ? await alertsAPI.getUnread()
        : await alertsAPI.getAll();
      

      let data = res.data.alerts || res.data || [];
      if (filter === 'read') {
        data = data.filter(a => a.is_read);
      }
      setAlerts(data);
    } catch (error) {
      toast.error('Failed to fetch alerts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, [filter]);

  const handleMarkAsRead = async (alert) => {
    try {
      await alertsAPI.markAsRead(alert.id || alert._id);
      toast.success('Alert marked as read');
      fetchAlerts();
    } catch (error) {
      toast.error('Failed to mark as read');
    }
  };

  const handleMarkAllAsRead = async () => {
    try {
      await alertsAPI.markAllAsRead();
      toast.success('All alerts marked as read');
      fetchAlerts();
    } catch (error) {
      toast.error('Failed to mark all as read');
    }
  };

  const handleDelete = async (alert) => {
    const id = alert.id || alert._id;
    if (!id) return;
    if (!confirm('Delete this alert permanently?')) return;
    try {
      await alertsAPI.delete(id);
      toast.success('Alert deleted');
      fetchAlerts();
    } catch (error) {
      toast.error('Failed to delete alert');
    }
  };

  const handleDeleteAll = async () => {
    if (!confirm('Are you sure you want to delete ALL alerts? This action cannot be undone.')) return;
    try {
      const res = await alertsAPI.deleteAll();
      toast.success(res.data.message || 'All alerts deleted');
      fetchAlerts();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to delete all alerts');
    }
  };

  const unreadCount = alerts.filter(a => !a.is_read).length;

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Alert Center</h1>
          <p className="text-gray-500">
            MDR contact alerts and notifications
            {unreadCount > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-danger-100 text-danger-700 rounded-full text-sm">
                {unreadCount} unread
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={fetchAlerts} className="btn-secondary">
            <RefreshCw className="h-4 w-4" />
          </button>
          {unreadCount > 0 && (
            <button onClick={handleMarkAllAsRead} className="btn-secondary flex items-center gap-2">
              <CheckCheck className="h-4 w-4" />
              Mark All Read
            </button>
          )}
          {isAdmin && alerts.length > 0 && (
            <button 
              onClick={handleDeleteAll} 
              className="btn-secondary flex items-center gap-2 text-red-600 hover:bg-red-50 border-red-200"
            >
              <Trash2 className="h-4 w-4" />
              Clear All Alerts
            </button>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <Filter className="h-5 w-5 text-gray-400" />
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'all' 
              ? 'bg-primary-100 text-primary-700' 
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          All Alerts
        </button>
        <button
          onClick={() => setFilter('unread')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'unread' 
              ? 'bg-primary-100 text-primary-700' 
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Unread
        </button>
        <button
          onClick={() => setFilter('read')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'read' 
              ? 'bg-primary-100 text-primary-700' 
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Read
        </button>
      </div>

      {/* Alerts List */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-600" />
        </div>
      ) : alerts.length === 0 ? (
        <div className="card text-center py-20">
          <Bell className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-700 mb-2">No alerts</h3>
          <p className="text-gray-500">
            {filter === 'unread' 
              ? "You're all caught up!" 
              : "No MDR contact alerts have been recorded yet"
            }
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div
              key={alert.id || alert._id}
              className={`card transition-all ${
                !alert.is_read 
                  ? alert.alert_type === 'mdr_marked' 
                    ? 'border-l-4 border-l-purple-500 bg-purple-50/30'
                    : 'border-l-4 border-l-danger-500 bg-danger-50/30' 
                  : 'hover:shadow-md'
              }`}
            >
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-full ${
                  alert.alert_type === 'mdr_marked'
                    ? (!alert.is_read ? 'bg-purple-100' : 'bg-gray-100')
                    : (!alert.is_read ? 'bg-danger-100' : 'bg-gray-100')
                }`}>
                  {alert.alert_type === 'mdr_marked' ? (
                    <Users className={`h-6 w-6 ${!alert.is_read ? 'text-purple-600' : 'text-gray-500'}`} />
                  ) : (
                    <AlertTriangle className={`h-6 w-6 ${!alert.is_read ? 'text-danger-600' : 'text-gray-500'}`} />
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      {alert.alert_type === 'mdr_marked' ? (
                        <>
                          <h3 className={`font-semibold ${!alert.is_read ? 'text-gray-900' : 'text-gray-700'}`}>
                            New MDR Patient Marked
                          </h3>
                          <p className="text-gray-600 mt-1">
                            <span className="font-medium text-purple-600">{alert.mdr_patient}</span>
                            {' was marked as MDR positive'}
                            {alert.pathogen_type && (
                              <span className="text-danger-600"> ({alert.pathogen_type})</span>
                            )}
                            {alert.past_contacts && alert.past_contacts.length > 0 && (
                              <span className="text-orange-600"> • {alert.past_contacts.length} past contacts</span>
                            )}
                          </p>
                        </>
                      ) : (
                        <>
                          <h3 className={`font-semibold ${!alert.is_read ? 'text-gray-900' : 'text-gray-700'}`}>
                            MDR Contact Detected
                          </h3>
                          <p className="text-gray-600 mt-1">
                            <span className="font-medium text-danger-600">{alert.mdr_patient}</span>
                            {' came in contact with '}
                            <span className="font-medium">{alert.contacted_person || alert.contact_name || 'Unknown'}</span>
                          </p>
                        </>
                      )}
                    </div>
                    <div className="text-right text-sm text-gray-500 whitespace-nowrap">
                      {(() => {
                        // Use timestamp (when alert was created/triggered)
                        const displayTime = alert.timestamp || alert.created_at;
                        if (!displayTime) return null;
                        return (
                          <>
                            <div>{format(new Date(displayTime), 'MMM d, HH:mm')}</div>
                            <div className="text-xs">
                              {formatDistanceToNow(new Date(displayTime), { addSuffix: true })}
                            </div>
                            {alert.alert_type === 'mdr_contact' && (
                              <div className="text-xs text-red-600 font-medium">Live Alert</div>
                            )}
                            {alert.alert_type === 'mdr_marked' && (
                              <div className="text-xs text-purple-600 font-medium">MDR Marked</div>
                            )}
                          </>
                        );
                      })()}
                    </div>
                  </div>

                  <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
                    {alert.alert_type === 'mdr_marked' && alert.marked_by && (
                      <span>Marked by: {alert.marked_by}</span>
                    )}
                    {alert.duration_seconds > 0 && (
                      <span>Duration: {Math.round(alert.duration_seconds)}s</span>
                    )}
                    {(alert.distance_meters !== undefined && alert.distance_meters !== null) ? (
                      <span className="flex items-center gap-1">
                        <Ruler className="h-3 w-3" />
                        {alert.distance_meters.toFixed(2)}m
                      </span>
                    ) : (alert.min_distance_meters !== undefined && alert.min_distance_meters !== null) && (
                      <span className="flex items-center gap-1">
                        <Ruler className="h-3 w-3" />
                        {alert.min_distance_meters.toFixed(2)}m (min)
                      </span>
                    )}
                    {alert.risk_percent !== undefined && alert.risk_percent > 0 && (
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        alert.risk_percent >= 40 
                          ? 'bg-red-100 text-red-800' 
                          : alert.risk_percent >= 20
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                      }`}>
                        Risk: {alert.risk_percent.toFixed(1)}%
                      </span>
                    )}
                    {alert.contact_count > 1 && (
                      <span className="px-2 py-0.5 rounded text-xs bg-blue-100 text-blue-800">
                        {alert.contact_count} interactions
                      </span>
                    )}
                    {alert.email_sent && (
                      <span className="flex items-center gap-1 text-green-600">
                        <Mail className="h-3 w-3" />
                        Email sent
                      </span>
                    )}
                    {(alert.snapshot_base64 || alert.has_front_snapshot || alert.has_side_snapshot) && (
                      <span className="flex items-center gap-1 text-primary-600">
                        <Image className="h-3 w-3" />
                        Has snapshot
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 mt-3">
                    <button
                      onClick={() => setSelectedAlert(alert)}
                      className="btn-secondary text-sm py-1 flex items-center gap-1"
                    >
                      <Eye className="h-4 w-4" />
                      View Details
                    </button>
                    {!alert.is_read && (
                      <button
                        onClick={() => handleMarkAsRead(alert)}
                        className="btn-secondary text-sm py-1 flex items-center gap-1"
                      >
                        <Check className="h-4 w-4" />
                        Mark Read
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(alert)}
                      className="btn-secondary text-sm py-1 flex items-center gap-1 text-red-600"
                    >
                      <Trash2 className="h-4 w-4" />
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto animate-fadeIn">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                {selectedAlert.alert_type === 'mdr_marked' ? (
                  <>
                    <Users className="h-5 w-5 text-purple-500" />
                    MDR Patient Marked
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-5 w-5 text-danger-500" />
                    Alert Details
                  </>
                )}
              </h3>
              <button 
                onClick={() => setSelectedAlert(null)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* MDR Marked Alert Content */}
              {selectedAlert.alert_type === 'mdr_marked' ? (
                <>
                  {/* Patient Info */}
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <p className="text-xs text-purple-600 uppercase font-medium mb-2">MDR Patient</p>
                    <div className="flex items-center gap-3">
                      <div className="h-12 w-12 rounded-full bg-purple-100 flex items-center justify-center">
                        <User className="h-6 w-6 text-purple-600" />
                      </div>
                      <div>
                        <span className="font-semibold text-lg text-gray-900">{selectedAlert.mdr_patient}</span>
                        {selectedAlert.pathogen_type && (
                          <p className="text-danger-600 font-medium">{selectedAlert.pathogen_type}</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Details */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Alert Type</span>
                      <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded font-medium text-sm">
                        MDR Patient Marked
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Marked At</span>
                      <span className="font-medium">
                        {(selectedAlert.timestamp || selectedAlert.created_at)
                          ? format(new Date(selectedAlert.timestamp || selectedAlert.created_at), 'MMMM d, yyyy HH:mm:ss')
                          : 'N/A'
                        }
                      </span>
                    </div>
                    {selectedAlert.marked_by && (
                      <div className="flex items-center justify-between py-2 border-b border-gray-100">
                        <span className="text-gray-500">Marked By</span>
                        <span className="font-medium">{selectedAlert.marked_by}</span>
                      </div>
                    )}
                    {selectedAlert.notes && (
                      <div className="py-2 border-b border-gray-100">
                        <span className="text-gray-500">Notes</span>
                        <p className="mt-1 text-gray-800">{selectedAlert.notes}</p>
                      </div>
                    )}
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Email Notification</span>
                      <span className={`flex items-center gap-1 ${
                        selectedAlert.email_sent ? 'text-green-600' : 'text-gray-400'
                      }`}>
                        <Mail className="h-4 w-4" />
                        {selectedAlert.email_sent ? 'Sent' : 'Not sent'}
                      </span>
                    </div>
                  </div>

                  {/* Past Contacts Section */}
                  {selectedAlert.past_contacts && selectedAlert.past_contacts.length > 0 && (
                    <div className="mt-6">
                      <h4 className="text-md font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Users className="h-5 w-5 text-orange-500" />
                        Past Contact History ({selectedAlert.past_contacts.length} contacts)
                      </h4>
                      <div className="bg-orange-50 border border-orange-200 rounded-lg overflow-hidden">
                        <table className="min-w-full divide-y divide-orange-200">
                          <thead className="bg-orange-100">
                            <tr>
                              <th className="px-4 py-2 text-left text-xs font-medium text-orange-800 uppercase">Contact Person</th>
                              <th className="px-4 py-2 text-left text-xs font-medium text-orange-800 uppercase">Contact Time</th>
                              <th className="px-4 py-2 text-left text-xs font-medium text-orange-800 uppercase">Duration</th>
                              <th className="px-4 py-2 text-left text-xs font-medium text-orange-800 uppercase">Risk</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-orange-100">
                            {selectedAlert.past_contacts.map((contact, index) => (
                              <tr key={index} className="hover:bg-orange-50">
                                <td className="px-4 py-2 whitespace-nowrap">
                                  <div className="flex items-center gap-2">
                                    <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                                      <User className="h-4 w-4 text-blue-600" />
                                    </div>
                                    <span className="font-medium text-gray-900">{contact.person_name || contact.contacted_person || contact.contact_name}</span>
                                  </div>
                                </td>
                                <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-600">
                                  {(contact.last_contact || contact.first_contact || contact.contact_time) 
                                    ? format(new Date(contact.last_contact || contact.first_contact || contact.contact_time), 'MMM d, HH:mm') 
                                    : 'N/A'}
                                </td>
                                <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-600">
                                  {contact.total_duration 
                                    ? `${Math.round(contact.total_duration)}s`
                                    : contact.duration_seconds 
                                      ? `${Math.round(contact.duration_seconds)}s`
                                      : contact.duration_minutes 
                                        ? `${contact.duration_minutes.toFixed(1)}min`
                                        : 'N/A'
                                  }
                                </td>
                                <td className="px-4 py-2 whitespace-nowrap">
                                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                                    (contact.risk_percent || 0) >= 40 
                                      ? 'bg-red-100 text-red-800' 
                                      : (contact.risk_percent || 0) >= 20
                                        ? 'bg-yellow-100 text-yellow-800'
                                        : 'bg-green-100 text-green-800'
                                  }`}>
                                    {(contact.risk_percent || 0).toFixed(1)}%
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* No Past Contacts Message */}
                  {(!selectedAlert.past_contacts || selectedAlert.past_contacts.length === 0) && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg text-center">
                      <Users className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-gray-500">No past contacts found during incubation period</p>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {/* Persons Involved - Contact Alert */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-red-50 rounded-lg">
                      <p className="text-xs text-red-600 uppercase font-medium mb-2">MDR Patient</p>
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-red-100 flex items-center justify-center">
                          <User className="h-5 w-5 text-red-600" />
                        </div>
                        <span className="font-medium text-gray-900">{selectedAlert.mdr_patient}</span>
                      </div>
                    </div>
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <p className="text-xs text-blue-600 uppercase font-medium mb-2">Contact Person</p>
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                          <User className="h-5 w-5 text-blue-600" />
                        </div>
                        <span className="font-medium text-gray-900">{selectedAlert.contacted_person || selectedAlert.contact_name || 'Unknown'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  <div className={`p-4 rounded-lg border-2 ${
                    (selectedAlert.risk_percent || 0) >= 40 
                      ? 'border-red-300 bg-red-50' 
                      : (selectedAlert.risk_percent || 0) >= 20 
                        ? 'border-yellow-300 bg-yellow-50'
                        : 'border-green-300 bg-green-50'
                  }`}>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Risk Factor</p>
                        <p className={`text-2xl font-bold ${
                          (selectedAlert.risk_percent || 0) >= 40 
                            ? 'text-red-600' 
                            : (selectedAlert.risk_percent || 0) >= 20 
                              ? 'text-yellow-600'
                              : 'text-green-600'
                        }`}>
                          {(selectedAlert.risk_percent || 0).toFixed(1)}%
                        </p>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        (selectedAlert.risk_percent || 0) >= 40 
                          ? 'bg-red-100 text-red-800' 
                          : (selectedAlert.risk_percent || 0) >= 20 
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                      }`}>
                        {(selectedAlert.risk_percent || 0) >= 40 ? 'High Risk' : (selectedAlert.risk_percent || 0) >= 20 ? 'Medium Risk' : 'Low Risk'}
                      </div>
                    </div>
                  </div>

                  {/* Details */}
                  <div className="space-y-3">
                    {selectedAlert.alert_type === 'mdr_contact' && (
                      <div className="flex items-center justify-between py-2 border-b border-gray-100">
                        <span className="text-gray-500">Alert Type</span>
                        <span className="px-2 py-1 bg-red-100 text-red-800 rounded font-medium text-sm">
                          Live Alert
                        </span>
                      </div>
                    )}
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Alert Time</span>
                      <span className="font-medium">
                        {(selectedAlert.timestamp || selectedAlert.created_at)
                          ? format(new Date(selectedAlert.timestamp || selectedAlert.created_at), 'MMMM d, yyyy HH:mm:ss')
                          : 'N/A'
                        }
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Contact Duration</span>
                      <span className="font-medium">
                        {selectedAlert.duration_seconds 
                          ? `${Math.round(selectedAlert.duration_seconds)} seconds` 
                          : 'N/A'
                        }
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500 flex items-center gap-1">
                        <Ruler className="h-4 w-4" />
                        Contact Distance
                      </span>
                      <span className="font-medium">
                        {selectedAlert.distance_meters !== undefined && selectedAlert.distance_meters !== null
                          ? `${selectedAlert.distance_meters.toFixed(2)} m`
                          : selectedAlert.min_distance_meters !== undefined && selectedAlert.min_distance_meters !== null
                            ? `${selectedAlert.min_distance_meters.toFixed(2)} m (min)`
                            : 'N/A'
                        }
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Total Interactions</span>
                      <span className="font-medium">
                        {selectedAlert.contact_count ? `${selectedAlert.contact_count} contacts` : '1 contact'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Email Notification</span>
                      <span className={`flex items-center gap-1 ${
                        selectedAlert.email_sent ? 'text-green-600' : 'text-gray-400'
                      }`}>
                        <Mail className="h-4 w-4" />
                        {selectedAlert.email_sent ? 'Sent' : 'Not sent'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-500">Status</span>
                      <span className={`flex items-center gap-1 ${
                        selectedAlert.is_read ? 'text-gray-600' : 'text-danger-600'
                      }`}>
                        {selectedAlert.is_read ? (
                          <>
                            <Check className="h-4 w-4" />
                            Read
                          </>
                        ) : (
                          <>
                            <Bell className="h-4 w-4" />
                            Unread
                          </>
                        )}
                      </span>
                    </div>
                  </div>
                </>
              )}

              {/* Snapshot */}
              {selectedAlert.snapshot_base64 && (
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Snapshot</p>
                  <img
                    src={selectedAlert.snapshot_base64.startsWith('data:') 
                      ? selectedAlert.snapshot_base64 
                      : `data:image/jpeg;base64,${selectedAlert.snapshot_base64}`
                    }
                    alt="Contact snapshot"
                    className="w-full rounded-lg border border-gray-200"
                  />
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                {!selectedAlert.is_read && (
                  <button
                    onClick={() => {
                      handleMarkAsRead(selectedAlert);
                      setSelectedAlert({ ...selectedAlert, is_read: true });
                    }}
                    className="btn-primary flex items-center gap-2"
                  >
                    <Check className="h-4 w-4" />
                    Mark as Read
                  </button>
                )}
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="btn-secondary ml-auto"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
