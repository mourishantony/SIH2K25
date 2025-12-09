import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { mdrAPI, personsAPI, pathogensAPI } from '../api';
import { useAuth } from '../context/AuthContext';
import { 
  AlertTriangle, Search, Plus, X, User, Check,
  Calendar, Users, ArrowRight, RefreshCw, Biohazard,
  Edit2, Trash2, Save, Bug
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';

export default function MDRManagement() {
  const { hasPermission } = useAuth();
  const [mdrPatients, setMdrPatients] = useState([]);
  const [eligiblePersons, setEligiblePersons] = useState([]);
  const [pathogens, setPathogens] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showMarkModal, setShowMarkModal] = useState(false);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [selectedPathogen, setSelectedPathogen] = useState('Other');
  const [mdrNotes, setMdrNotes] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedContacts, setSelectedContacts] = useState(null);
  
  const [showPathogenModal, setShowPathogenModal] = useState(false);
  const [pathogenForm, setPathogenForm] = useState({ name: '', risk_factor: 1.0, incubation_days: 14, description: '' });
  const [editingPathogen, setEditingPathogen] = useState(null);
  const [activeTab, setActiveTab] = useState('patients'); 

  const fetchData = async () => {
    try {
      setLoading(true);
      const [mdrRes, eligibleRes, pathogensRes] = await Promise.all([
        mdrAPI.getPatients(),
        mdrAPI.getEligible(),
        pathogensAPI.getAll()
      ]);
      setMdrPatients(mdrRes.data.patients || mdrRes.data || []);
      setEligiblePersons(eligibleRes.data.eligible_patients || eligibleRes.data || []);
      setPathogens(pathogensRes.data || []);
    } catch (error) {
      toast.error('Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleMarkAsMDR = async () => {
    if (!selectedPerson) return;
    
    try {
      await mdrAPI.markAsMDR(selectedPerson, selectedPathogen, mdrNotes);
      toast.success(`${selectedPerson} marked as MDR patient (${selectedPathogen})`);
      setShowMarkModal(false);
      setSelectedPerson(null);
      setSelectedPathogen('Other');
      setMdrNotes('');
      fetchData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to mark as MDR');
    }
  };

  const handleUnmarkMDR = async (name) => {
    try {
      await mdrAPI.unmarkMDR(name);
      toast.success(`${name} removed from MDR list`);
      fetchData();
    } catch (error) {
      toast.error('Failed to unmark MDR patient');
    }
  };

  const viewContacts = async (name) => {
    try {
      const res = await mdrAPI.getContacts(name);
     
      setSelectedContacts({ name, contacts: res.data.contacts || res.data || [] });
    } catch (error) {
      toast.error('Failed to fetch contacts');
    }
  };

  const handleSavePathogen = async () => {
    try {
      if (editingPathogen) {
        await pathogensAPI.update(editingPathogen, {
          risk_factor: parseFloat(pathogenForm.risk_factor),
          incubation_days: parseInt(pathogenForm.incubation_days),
          description: pathogenForm.description,
        });
        toast.success(`Pathogen "${editingPathogen}" updated`);
      } else {
        await pathogensAPI.create({
          name: pathogenForm.name,
          risk_factor: parseFloat(pathogenForm.risk_factor),
          incubation_days: parseInt(pathogenForm.incubation_days),
          description: pathogenForm.description,
        });
        toast.success(`Pathogen "${pathogenForm.name}" created`);
      }
      setShowPathogenModal(false);
      setPathogenForm({ name: '', risk_factor: 1.0, incubation_days: 14, description: '' });
      setEditingPathogen(null);
      fetchData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to save pathogen');
    }
  };

  const handleEditPathogen = (pathogen) => {
    setEditingPathogen(pathogen.name);
    setPathogenForm({
      name: pathogen.name,
      risk_factor: pathogen.risk_factor,
      incubation_days: pathogen.incubation_days,
      description: pathogen.description || '',
    });
    setShowPathogenModal(true);
  };

  const handleDeletePathogen = async (name) => {
    if (!confirm(`Delete pathogen "${name}"? This cannot be undone.`)) return;
    try {
      await pathogensAPI.delete(name);
      toast.success(`Pathogen "${name}" deleted`);
      fetchData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to delete pathogen');
    }
  };

  const filteredEligible = eligiblePersons.filter(p => 
    p.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getPathogenColor = (type) => {
    const colors = {
      'MRSA': 'bg-orange-100 text-orange-700 border-orange-200',
      'MDR-TB': 'bg-red-100 text-red-700 border-red-200',
      'VRE': 'bg-purple-100 text-purple-700 border-purple-200',
      'CRE': 'bg-pink-100 text-pink-700 border-pink-200',
      'ESBL': 'bg-yellow-100 text-yellow-700 border-yellow-200',
      'Other': 'bg-gray-100 text-gray-700 border-gray-200',
    };
    return colors[type] || colors['Other'];
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">MDR Management</h1>
          <p className="text-gray-500">Manage MDR patients and pathogen types</p>
        </div>
        <div className="flex gap-2">
          {hasPermission('pathogen_management') && (
            <button 
              onClick={() => {
                setEditingPathogen(null);
                setPathogenForm({ name: '', risk_factor: 1.0, incubation_days: 14, description: '' });
                setShowPathogenModal(true);
              }}
              className="btn-secondary flex items-center gap-2"
            >
              <Bug className="h-4 w-4" />
              Add Pathogen
            </button>
          )}
          <button 
            onClick={() => setShowMarkModal(true)}
            className="btn-danger flex items-center gap-2"
          >
            <AlertTriangle className="h-4 w-4" />
            Mark as MDR
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-4">
          <button
            onClick={() => setActiveTab('patients')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'patients'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            MDR Patients ({mdrPatients.length})
          </button>
          {hasPermission('pathogen_management') && (
            <button
              onClick={() => setActiveTab('pathogens')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'pathogens'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Pathogen Types ({pathogens.length})
            </button>
          )}
        </nav>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-red-50 border-red-100">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-red-100 rounded-full">
              <AlertTriangle className="h-6 w-6 text-red-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-red-700">{mdrPatients.length}</p>
              <p className="text-red-600 text-sm">Active MDR Patients</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gray-100 rounded-full">
              <Users className="h-6 w-6 text-gray-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-700">{eligiblePersons.length}</p>
              <p className="text-gray-500 text-sm">Registered Patients</p>
            </div>
          </div>
        </div>
        <div className="card">
          <Link to="/alerts" className="flex items-center gap-4 hover:opacity-80">
            <div className="p-3 bg-yellow-100 rounded-full">
              <AlertTriangle className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="flex-1">
              <p className="text-gray-700 font-medium">View Contact Alerts</p>
              <p className="text-gray-500 text-sm">See all MDR contact incidents</p>
            </div>
            <ArrowRight className="h-5 w-5 text-gray-400" />
          </Link>
        </div>
      </div>

      {/* MDR Patients List - shown when patients tab active */}
      {activeTab === 'patients' && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              MDR Patients
            </h2>
            <button onClick={fetchData} className="btn-secondary p-2">
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
            </div>
          ) : mdrPatients.length === 0 ? (
            <div className="text-center py-12">
              <Check className="h-12 w-12 text-green-400 mx-auto mb-4" />
              <p className="text-gray-500">No MDR patients currently marked</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {mdrPatients.map((patient, index) => (
                <div key={index} className="py-4 flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="h-12 w-12 rounded-full bg-red-100 flex items-center justify-center">
                      <User className="h-6 w-6 text-red-600" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium text-gray-900">{patient.name}</h3>
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${getPathogenColor(patient.pathogen_type)}`}>
                          {patient.pathogen_type || 'Other'} ({patient.pathogen_factor || 1.0})
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          Marked: {patient.marked_at ? format(new Date(patient.marked_at), 'MMM d, yyyy') : 'N/A'}
                        </span>
                        {patient.contact_count !== undefined && (
                          <span className="flex items-center gap-1">
                            <Users className="h-3 w-3" />
                            {patient.contact_count} contacts
                          </span>
                        )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => viewContacts(patient.name)}
                    className="btn-secondary text-sm py-1"
                  >
                    View Contacts
                  </button>
                  <button
                    onClick={() => handleUnmarkMDR(patient.name)}
                    className="btn-secondary text-sm py-1 text-red-600 hover:bg-red-50"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
        </div>
      )}

      {/* Pathogen Types List - shown when pathogens tab active */}
      {activeTab === 'pathogens' && hasPermission('pathogen_management') && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Bug className="h-5 w-5 text-purple-500" />
              MDR Pathogen Types
            </h2>
            <button onClick={fetchData} className="btn-secondary p-2">
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
            </div>
          ) : pathogens.length === 0 ? (
            <div className="text-center py-12">
              <Bug className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No pathogen types defined</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Pathogen Name</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk Factor</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Incubation Period</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Description</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {pathogens.map((pathogen) => (
                    <tr key={pathogen.name} className="hover:bg-gray-50">
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 text-sm font-medium rounded-full border ${getPathogenColor(pathogen.name)}`}>
                          {pathogen.name}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {pathogen.risk_factor}x
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {pathogen.incubation_days} days
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500">
                        {pathogen.description || '-'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => handleEditPathogen(pathogen)}
                            className="p-1 hover:bg-gray-100 rounded text-gray-600"
                            title="Edit"
                          >
                            <Edit2 className="h-4 w-4" />
                          </button>
                          {pathogen.name !== 'Other' && (
                            <button
                              onClick={() => handleDeletePathogen(pathogen.name)}
                              className="p-1 hover:bg-red-50 rounded text-red-600"
                              title="Delete"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Mark as MDR Modal */}
      {showMarkModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 animate-fadeIn">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Mark Patient as MDR
            </h3>
            
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search patients..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input pl-10"
                />
              </div>
            </div>

            <div className="max-h-48 overflow-y-auto mb-4">
              {filteredEligible.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No eligible patients found
                </div>
              ) : (
                <div className="space-y-2">
                  {filteredEligible.map((person) => (
                    <button
                      key={person._id}
                      onClick={() => setSelectedPerson(person.name)}
                      className={`w-full p-3 rounded-lg border text-left transition-colors ${
                        selectedPerson === person.name
                          ? 'border-red-500 bg-red-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center">
                          <User className="h-4 w-4 text-gray-600" />
                        </div>
                        <div className="flex-1">
                          <p className="font-medium text-gray-900">{person.name}</p>
                          <p className="text-xs text-gray-500">{person.role}</p>
                        </div>
                        {selectedPerson === person.name && (
                          <Check className="h-5 w-5 text-red-500" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Pathogen Type Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Pathogen Type (Risk Factor)
              </label>
              <select
                value={selectedPathogen}
                onChange={(e) => setSelectedPathogen(e.target.value)}
                className="input"
              >
                {pathogens.map((p) => (
                  <option key={p.type} value={p.type}>
                    {p.type} - {p.description} (Factor: {p.factor})
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Higher factor = more dangerous pathogen (used in risk calculation)
              </p>
            </div>

            {/* Notes */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Notes (Optional)
              </label>
              <textarea
                value={mdrNotes}
                onChange={(e) => setMdrNotes(e.target.value)}
                placeholder="Additional notes about the MDR status..."
                className="input min-h-[60px]"
                rows={2}
              />
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
              <p className="text-sm text-yellow-800">
                <strong>Warning:</strong> Marking a patient as MDR will trigger alerts for all 
                detected contacts.
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <button 
                onClick={() => {
                  setShowMarkModal(false);
                  setSelectedPerson(null);
                  setSelectedPathogen('Other');
                  setMdrNotes('');
                  setSearchQuery('');
                }}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button 
                onClick={handleMarkAsMDR}
                disabled={!selectedPerson}
                className="btn-danger"
              >
                Mark as MDR
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Contacts Modal */}
      {selectedContacts && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-2xl w-full mx-4 animate-fadeIn">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">
                Contacts for {selectedContacts.name}
              </h3>
              <button onClick={() => setSelectedContacts(null)} className="p-1 hover:bg-gray-100 rounded">
                <X className="h-5 w-5" />
              </button>
            </div>

            {selectedContacts.contacts.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                No contacts recorded yet
              </div>
            ) : (
              <div className="max-h-96 overflow-y-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Contact Person
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Last Contact
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Duration
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Distance
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Risk %
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Interactions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {selectedContacts.contacts.map((contact, idx) => (
                      <tr key={idx}>
                        <td className="px-4 py-3">
                          <span className="font-medium">{contact.contact_name || contact.other_person || 'Unknown'}</span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-500">
                          {contact.timestamp || contact.last_contact
                            ? format(new Date(contact.timestamp || contact.last_contact), 'MMM d, HH:mm')
                            : 'N/A'
                          }
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-500">
                          {contact.duration_seconds ? `${Math.round(contact.duration_seconds)}s` : 'N/A'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-500">
                          {contact.min_distance_meters !== undefined && contact.min_distance_meters !== null
                            ? `${contact.min_distance_meters.toFixed(2)}m`
                            : contact.distance_meters !== undefined && contact.distance_meters !== null
                              ? `${contact.distance_meters.toFixed(2)}m`
                              : 'N/A'}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            (contact.max_risk_percent || 0) >= 40 
                              ? 'bg-red-100 text-red-800' 
                              : (contact.max_risk_percent || 0) >= 20
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-green-100 text-green-800'
                          }`}>
                            {(contact.max_risk_percent || 0).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-600">
                          {contact.contact_count ? `${contact.contact_count} contacts` : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="flex justify-end mt-4">
              <button onClick={() => setSelectedContacts(null)} className="btn-secondary">
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Pathogen Add/Edit Modal */}
      {showPathogenModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 animate-fadeIn">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Bug className="h-5 w-5 text-purple-500" />
              {editingPathogen ? 'Edit Pathogen' : 'Add New Pathogen'}
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Pathogen Name
                </label>
                <input
                  type="text"
                  value={pathogenForm.name}
                  onChange={(e) => setPathogenForm({ ...pathogenForm, name: e.target.value })}
                  disabled={!!editingPathogen}
                  placeholder="e.g., MRSA, VRE, CRE"
                  className="input disabled:bg-gray-100"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Risk Factor
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="5"
                  value={pathogenForm.risk_factor}
                  onChange={(e) => setPathogenForm({ ...pathogenForm, risk_factor: parseFloat(e.target.value) || 1.0 })}
                  className="input"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Multiplier for risk calculation (1.0 = baseline, higher = more dangerous)
                </p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Incubation Period (days)
                </label>
                <input
                  type="number"
                  step="1"
                  min="1"
                  max="365"
                  value={pathogenForm.incubation_days}
                  onChange={(e) => setPathogenForm({ ...pathogenForm, incubation_days: parseInt(e.target.value) || 14 })}
                  className="input"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Time window to track contacts after exposure
                </p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={pathogenForm.description}
                  onChange={(e) => setPathogenForm({ ...pathogenForm, description: e.target.value })}
                  placeholder="Brief description of the pathogen"
                  rows={2}
                  className="input"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => {
                  setShowPathogenModal(false);
                  setEditingPathogen(null);
                  setPathogenForm({ name: '', risk_factor: 1.0, incubation_days: 14, description: '' });
                }}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleSavePathogen}
                disabled={!editingPathogen && !pathogenForm.name.trim()}
                className="btn-primary flex items-center gap-2"
              >
                <Save className="h-4 w-4" />
                {editingPathogen ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
