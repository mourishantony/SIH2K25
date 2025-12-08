import { useState, useEffect } from 'react';
import { 
  Users, UserPlus, Edit2, Trash2, Save, X, Shield, 
  Loader2, AlertCircle, CheckCircle
} from 'lucide-react';
import { usersAPI } from '../api';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

const ROLE_LABELS = {
  admin: { label: 'Administrator', color: 'bg-purple-100 text-purple-700' },
  ehr_user: { label: 'EHR System User', color: 'bg-blue-100 text-blue-700' },
  officer: { label: 'Officer', color: 'bg-green-100 text-green-700' },
};

const ROLE_OPTIONS = [
  { value: 'ehr_user', label: 'EHR System User', description: 'Dashboard, Registered Persons, MDR Management, Alerts' },
  { value: 'officer', label: 'Officer', description: 'Dashboard, Register Person, Registered Persons, Unknown Persons' },
];

export default function UserManagement() {
  const { user: currentUser, isAdmin } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [saving, setSaving] = useState(false);

  const getUserId = (user) => user?.id || user?._id;

  // Form state
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    role: 'officer',
  });

  useEffect(() => {
    if (isAdmin) {
      fetchUsers();
    }
  }, [isAdmin]);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const response = await usersAPI.getAll();
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
      toast.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateUser = async (e) => {
    e.preventDefault();
    
    if (!formData.username || !formData.email || !formData.password) {
      toast.error('Please fill in all required fields');
      return;
    }

    setSaving(true);
    try {
      await usersAPI.create(formData);
      toast.success('User created successfully');
      setShowCreateModal(false);
      resetForm();
      fetchUsers();
    } catch (error) {
      console.error('Create user error:', error);
      toast.error(error.response?.data?.detail || 'Failed to create user');
    } finally {
      setSaving(false);
    }
  };

  const handleUpdateUser = async (e) => {
    e.preventDefault();
    
    if (!editingUser) return;

    const updateData = {
      email: formData.email,
      role: formData.role,
    };

    // Only include password if it was changed
    if (formData.password) {
      updateData.password = formData.password;
    }

    setSaving(true);
    try {
      await usersAPI.update(getUserId(editingUser), updateData);
      toast.success('User updated successfully');
      setEditingUser(null);
      resetForm();
      fetchUsers();
    } catch (error) {
      console.error('Update user error:', error);
      toast.error(error.response?.data?.detail || 'Failed to update user');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteUser = async (userId, username) => {
    if (username === currentUser?.username) {
      toast.error('You cannot delete your own account');
      return;
    }

    if (!confirm(`Are you sure you want to delete user "${username}"? This action cannot be undone.`)) {
      return;
    }

    try {
      await usersAPI.delete(userId);
      toast.success('User deleted successfully');
      fetchUsers();
    } catch (error) {
      console.error('Delete user error:', error);
      toast.error(error.response?.data?.detail || 'Failed to delete user');
    }
  };

  const openEditModal = (user) => {
    setEditingUser(user);
    setFormData({
      username: user.username,
      email: user.email,
      password: '',
      role: user.role,
    });
  };

  const resetForm = () => {
    setFormData({
      username: '',
      email: '',
      password: '',
      role: 'officer',
    });
  };

  if (!isAdmin) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-800 mb-2">Access Denied</h2>
          <p className="text-gray-600">You don't have permission to access this page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">User Management</h1>
          <p className="text-gray-600">Manage system users and their roles</p>
        </div>
        <button
          onClick={() => {
            resetForm();
            setShowCreateModal(true);
          }}
          className="btn btn-primary"
        >
          <UserPlus className="h-4 w-4" />
          Add User
        </button>
      </div>

      {/* Role Legend */}
      <div className="card">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Role Permissions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {ROLE_OPTIONS.map(role => (
            <div key={role.value} className="p-3 bg-gray-50 rounded-lg">
              <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${ROLE_LABELS[role.value].color}`}>
                {ROLE_LABELS[role.value].label}
              </span>
              <p className="text-sm text-gray-600 mt-2">
                {role.description || role.label}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Users Table */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
          <Users className="h-5 w-5 text-primary-600" />
          System Users ({users.length})
        </h3>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary-600" />
          </div>
        ) : users.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No users found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Username</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Email</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Role</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Created</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {users.map((user) => (
                    <tr key={getUserId(user)} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                          <span className="text-primary-700 font-medium text-sm">
                            {user.username?.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <span className="font-medium text-gray-800">
                          {user.username}
                          {user.username === currentUser?.username && (
                            <span className="ml-2 text-xs text-gray-500">(You)</span>
                          )}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-600">{user.email}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${ROLE_LABELS[user.role]?.color || 'bg-gray-100 text-gray-700'}`}>
                        {ROLE_LABELS[user.role]?.label || user.role}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-600 text-sm">
                      {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => openEditModal(user)}
                          className="p-2 text-gray-400 hover:text-primary-600 transition-colors"
                          title="Edit user"
                        >
                          <Edit2 className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteUser(getUserId(user), user.username)}
                          disabled={user.username === currentUser?.username}
                          className="p-2 text-gray-400 hover:text-red-600 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                          title={user.username === currentUser?.username ? "Can't delete yourself" : "Delete user"}
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

      {/* Create User Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 animate-fadeIn">
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="text-lg font-medium flex items-center gap-2">
                <UserPlus className="h-5 w-5 text-primary-600" />
                Create New User
              </h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleCreateUser} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Username *
                </label>
                <input
                  type="text"
                  value={formData.username}
                  onChange={(e) => setFormData(prev => ({ ...prev, username: e.target.value }))}
                  className="input-field"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email *
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  className="input-field"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Password *
                </label>
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData(prev => ({ ...prev, password: e.target.value }))}
                  className="input-field"
                  required
                  minLength={6}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Role *
                </label>
                <select
                  value={formData.role}
                  onChange={(e) => setFormData(prev => ({ ...prev, role: e.target.value }))}
                  className="input-field"
                >
                  {ROLE_OPTIONS.map(role => (
                    <option key={role.value} value={role.value}>
                      {role.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex justify-end gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="btn btn-secondary"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="btn btn-primary"
                >
                  {saving ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-4 w-4" />
                      Create User
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {editingUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 animate-fadeIn">
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="text-lg font-medium flex items-center gap-2">
                <Edit2 className="h-5 w-5 text-primary-600" />
                Edit User: {editingUser.username}
              </h3>
              <button
                onClick={() => setEditingUser(null)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleUpdateUser} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Username
                </label>
                <input
                  type="text"
                  value={formData.username}
                  className="input-field bg-gray-100"
                  disabled
                />
                <p className="text-xs text-gray-500 mt-1">Username cannot be changed</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  className="input-field"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  New Password (leave blank to keep current)
                </label>
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData(prev => ({ ...prev, password: e.target.value }))}
                  className="input-field"
                  placeholder="Enter new password..."
                  minLength={6}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Role
                </label>
                <select
                  value={formData.role}
                  onChange={(e) => setFormData(prev => ({ ...prev, role: e.target.value }))}
                  className="input-field"
                  disabled={editingUser.username === currentUser?.username}
                >
                  {ROLE_OPTIONS.map(role => (
                    <option key={role.value} value={role.value}>
                      {role.label}
                    </option>
                  ))}
                </select>
                {editingUser.username === currentUser?.username && (
                  <p className="text-xs text-gray-500 mt-1">You cannot change your own role</p>
                )}
              </div>
              <div className="flex justify-end gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setEditingUser(null)}
                  className="btn btn-secondary"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="btn btn-primary"
                >
                  {saving ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="h-4 w-4" />
                      Save Changes
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
