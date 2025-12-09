import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { personsAPI, faceAPI } from '../api';
import { useAuth } from '../context/AuthContext';
import { 
  Search, Filter, Plus, Edit, Trash2, User, 
  Phone, MapPin, Check, X, RefreshCw, Eye
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';

export default function PersonsList() {
  const { hasPermission } = useAuth();
  const [persons, setPersons] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [editingPerson, setEditingPerson] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState({});

  const fetchPersons = async () => {
    try {
      setLoading(true);
      const params = {};
      if (roleFilter) params.role = roleFilter;
      if (searchQuery) params.search = searchQuery;
      
      const response = await personsAPI.getAll(params);
      
      const personsList = Array.isArray(response.data) ? response.data : (response.data.persons || []);
      setPersons(personsList);
      
      
      const statusPromises = personsList.map(p => 
        faceAPI.getTrainingStatus(p.name).catch(() => ({ data: { is_trained: false } }))
      );
      const statuses = await Promise.all(statusPromises);
      const statusMap = {};
      personsList.forEach((p, i) => {
        statusMap[p.name] = statuses[i].data;
      });
      setTrainingStatus(statusMap);
      
    } catch (error) {
      toast.error('Failed to fetch persons');
      console.error('Fetch persons error:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPersons();
  }, [roleFilter, searchQuery]);

  const handleDelete = async (person) => {
    try {
      await personsAPI.delete(person.id);
      toast.success(`${person.name} deleted successfully`);
      setShowDeleteModal(null);
      fetchPersons();
    } catch (error) {
      toast.error('Failed to delete person');
    }
  };

  const handleUpdate = async (e) => {
    e.preventDefault();
    try {
      await personsAPI.update(editingPerson.id, editingPerson);
      toast.success('Person updated successfully');
      setEditingPerson(null);
      fetchPersons();
    } catch (error) {
      toast.error('Failed to update person');
    }
  };

  const handleRetrain = async (name) => {
    try {
      toast.loading(`Retraining ${name}...`, { id: 'retrain' });
      await faceAPI.retrain(name);
      toast.success(`Training started for ${name}`, { id: 'retrain' });
    } catch (error) {
      toast.error('Failed to start training', { id: 'retrain' });
    }
  };

  const getRoleColor = (role) => {
    const colors = {
      patient: 'bg-blue-100 text-blue-800',
      doctor: 'bg-green-100 text-green-800',
      visitor: 'bg-yellow-100 text-yellow-800',
      nurse: 'bg-pink-100 text-pink-800',
      worker: 'bg-purple-100 text-purple-800'
    };
    return colors[role] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Registered Persons</h1>
          <p className="text-gray-500">Manage patients, doctors, visitors, and workers</p>
        </div>
{hasPermission('register_person') && (
          <Link to="/register" className="btn-primary flex items-center gap-2">
            <Plus className="h-4 w-4" />
            Register Person
          </Link>
        )}
      </div>

      {/* Filters */}
      <div className="card flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search by name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input pl-10"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="h-5 w-5 text-gray-400" />
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            className="input w-40"
          >
            <option value="">All Roles</option>
            <option value="patient">Patients</option>
            <option value="doctor">Doctors</option>
            <option value="visitor">Visitors</option>
            <option value="nurse">Nurses</option>
            <option value="worker">Workers</option>
          </select>
        </div>
        <button onClick={fetchPersons} className="btn-secondary">
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>

      {/* Persons Table */}
      <div className="card overflow-hidden p-0">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-600" />
          </div>
        ) : persons.length === 0 ? (
          <div className="text-center py-20">
            <User className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">No persons found</p>
            {hasPermission('register_person') && (
              <Link to="/register" className="text-primary-600 hover:underline text-sm">
                Register a new person
              </Link>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Role
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Contact
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Face Trained
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Registered
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {persons.map((person) => (
                  <tr key={person._id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded text-sm font-mono font-bold">
                        {person.person_id || '-'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                          <User className="h-5 w-5 text-primary-600" />
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">{person.name}</div>
                          {person.notes && (
                            <div className="text-sm text-gray-500 truncate max-w-[200px]">{person.notes}</div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRoleColor(person.role)}`}>
                        {person.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {person.phone && (
                        <div className="flex items-center gap-1">
                          <Phone className="h-3 w-3" />
                          {person.phone}
                        </div>
                      )}
                      {person.place && (
                        <div className="flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          {person.place}
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {trainingStatus[person.name]?.is_trained ? (
                        <span className="flex items-center gap-1 text-green-600 text-sm">
                          <Check className="h-4 w-4" />
                          Trained ({trainingStatus[person.name]?.embedding_count || 0})
                        </span>
                      ) : trainingStatus[person.name]?.image_count > 0 ? (
                        <button
                          onClick={() => handleRetrain(person.name)}
                          className="flex items-center gap-1 text-yellow-600 text-sm hover:underline"
                        >
                          <RefreshCw className="h-4 w-4" />
                          Pending ({trainingStatus[person.name]?.image_count} images)
                        </button>
                      ) : (
                        <span className="flex items-center gap-1 text-gray-400 text-sm">
                          <X className="h-4 w-4" />
                          Not trained
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {person.created_at 
                        ? format(new Date(person.created_at), 'MMM d, yyyy')
                        : 'N/A'
                      }
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => setEditingPerson(person)}
                          className="p-1 text-gray-500 hover:text-primary-600"
                          title="Edit"
                        >
                          <Edit className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => setShowDeleteModal(person)}
                          className="p-1 text-gray-500 hover:text-danger-600"
                          title="Delete"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Edit Modal */}
      {editingPerson && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 animate-fadeIn">
            <h3 className="text-lg font-semibold mb-4">Edit Person</h3>
            <form onSubmit={handleUpdate} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={editingPerson.name}
                  onChange={(e) => setEditingPerson({ ...editingPerson, name: e.target.value })}
                  className="input"
                  disabled
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select
                  value={editingPerson.role}
                  onChange={(e) => setEditingPerson({ ...editingPerson, role: e.target.value })}
                  className="input"
                >
                  <option value="patient">Patient</option>
                  <option value="doctor">Doctor</option>
                  <option value="visitor">Visitor</option>
                  <option value="worker">Worker</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                <input
                  type="tel"
                  value={editingPerson.phone || ''}
                  onChange={(e) => setEditingPerson({ ...editingPerson, phone: e.target.value })}
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Place</label>
                <input
                  type="text"
                  value={editingPerson.place || ''}
                  onChange={(e) => setEditingPerson({ ...editingPerson, place: e.target.value })}
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                <textarea
                  value={editingPerson.notes || ''}
                  onChange={(e) => setEditingPerson({ ...editingPerson, notes: e.target.value })}
                  className="input"
                  rows={3}
                />
              </div>
              <div className="flex justify-end gap-3">
                <button type="button" onClick={() => setEditingPerson(null)} className="btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Save Changes
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-sm w-full mx-4 animate-fadeIn">
            <h3 className="text-lg font-semibold mb-2">Delete Person</h3>
            <p className="text-gray-600 mb-6">
              Are you sure you want to delete <strong>{showDeleteModal.name}</strong>? 
              This will also remove their face data and cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button onClick={() => setShowDeleteModal(null)} className="btn-secondary">
                Cancel
              </button>
              <button onClick={() => handleDelete(showDeleteModal)} className="btn-danger">
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
