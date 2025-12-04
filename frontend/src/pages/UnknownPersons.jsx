import { useState, useEffect } from 'react';
import { unknownAPI, personsAPI } from '../api';
import { 
  User, UserX, Camera, Clock, AlertTriangle, 
  RefreshCw, Eye, Search, Users, Shield, Link2
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format, formatDistanceToNow } from 'date-fns';

export default function UnknownPersons() {
  const [unknowns, setUnknowns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [contacts, setContacts] = useState([]);
  const [loadingContacts, setLoadingContacts] = useState(false);
  const [showLinkModal, setShowLinkModal] = useState(false);
  const [linkTarget, setLinkTarget] = useState(null);
  const [registeredPersons, setRegisteredPersons] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('all'); // all, mdr_contact, recent

  const fetchUnknowns = async () => {
    try {
      setLoading(true);
      const res = await unknownAPI.getAll();
      let data = res.data.unknown_persons || res.data || [];
      
      // Apply filters
      if (filter === 'mdr_contact') {
        data = data.filter(u => u.contacted_mdr);
      } else if (filter === 'recent') {
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        data = data.filter(u => new Date(u.last_seen) > oneDayAgo);
      }
      
      setUnknowns(data);
    } catch (error) {
      toast.error('Failed to fetch unknown persons');
    } finally {
      setLoading(false);
    }
  };

  const fetchContacts = async (tempId) => {
    try {
      setLoadingContacts(true);
      const res = await unknownAPI.getContacts(tempId);
      setContacts(res.data.contacts || res.data || []);
    } catch (error) {
      toast.error('Failed to fetch contacts');
    } finally {
      setLoadingContacts(false);
    }
  };

  const fetchRegisteredPersons = async () => {
    try {
      const res = await personsAPI.getAll();
      setRegisteredPersons(res.data.persons || res.data || []);
    } catch (error) {
      console.error('Failed to fetch registered persons');
    }
  };

  useEffect(() => {
    fetchUnknowns();
  }, [filter]);

  const handleViewContacts = async (person) => {
    setSelectedPerson(person);
    await fetchContacts(person.temp_id);
  };

  const handleLinkToPerson = (person) => {
    setLinkTarget(person);
    fetchRegisteredPersons();
    setShowLinkModal(true);
  };

  const handleConfirmLink = async (personName) => {
    try {
      await unknownAPI.markAsKnown(linkTarget.temp_id, personName);
      toast.success(`Linked ${linkTarget.temp_id} to ${personName}`);
      setShowLinkModal(false);
      setLinkTarget(null);
      fetchUnknowns();
    } catch (error) {
      toast.error('Failed to link person');
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    try {
      return format(new Date(dateStr), 'MMM dd, yyyy HH:mm');
    } catch {
      return dateStr;
    }
  };

  const filteredPersons = registeredPersons.filter(p => 
    p.name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const mdrContactCount = unknowns.filter(u => u.contacted_mdr).length;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <UserX className="h-7 w-7 text-warning-600" />
            Unknown Persons
          </h1>
          <p className="text-gray-500">
            Unregistered persons detected in contact with patients
            {mdrContactCount > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-danger-100 text-danger-700 rounded-full text-sm">
                {mdrContactCount} with MDR contact
              </span>
            )}
          </p>
        </div>
        <button onClick={fetchUnknowns} className="btn-secondary">
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'all' 
              ? 'bg-primary-100 text-primary-700' 
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          All ({unknowns.length})
        </button>
        <button
          onClick={() => setFilter('mdr_contact')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'mdr_contact' 
              ? 'bg-danger-100 text-danger-700' 
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          MDR Contact
        </button>
        <button
          onClick={() => setFilter('recent')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'recent' 
              ? 'bg-secondary-100 text-secondary-700' 
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Last 24h
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card p-4 flex items-center gap-3">
          <div className="p-3 bg-warning-100 rounded-lg">
            <UserX className="h-6 w-6 text-warning-600" />
          </div>
          <div>
            <p className="text-sm text-gray-500">Total Unknown</p>
            <p className="text-2xl font-bold text-gray-800">{unknowns.length}</p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <div className="p-3 bg-danger-100 rounded-lg">
            <AlertTriangle className="h-6 w-6 text-danger-600" />
          </div>
          <div>
            <p className="text-sm text-gray-500">MDR Contact</p>
            <p className="text-2xl font-bold text-danger-600">{mdrContactCount}</p>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3">
          <div className="p-3 bg-primary-100 rounded-lg">
            <Users className="h-6 w-6 text-primary-600" />
          </div>
          <div>
            <p className="text-sm text-gray-500">Total Contacts</p>
            <p className="text-2xl font-bold text-gray-800">
              {unknowns.reduce((acc, u) => acc + (u.contact_count || 0), 0)}
            </p>
          </div>
        </div>
      </div>

      {/* Unknown Persons Grid */}
      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        </div>
      ) : unknowns.length === 0 ? (
        <div className="card p-12 text-center">
          <UserX className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-500">No unknown persons detected</h3>
          <p className="text-gray-400 mt-2">
            Unknown persons will appear here when detected in contact with registered patients
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {unknowns.map((person) => (
            <div 
              key={person.temp_id} 
              className={`card p-4 hover:shadow-lg transition-shadow ${
                person.contacted_mdr ? 'border-l-4 border-l-danger-500' : ''
              }`}
            >
              <div className="flex items-start gap-4">
                {/* Face Snapshot */}
                <div className="w-20 h-20 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0">
                  {person.snapshot ? (
                    <img 
                      src={`data:image/jpeg;base64,${person.snapshot}`}
                      alt={person.temp_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <User className="h-10 w-10 text-gray-300" />
                    </div>
                  )}
                </div>
                
                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-gray-800 truncate">
                      {person.temp_id}
                    </h3>
                    {person.contacted_mdr && (
                      <span className="px-2 py-0.5 bg-danger-100 text-danger-700 rounded text-xs">
                        MDR
                      </span>
                    )}
                  </div>
                  
                  <div className="mt-2 space-y-1 text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      <span>First: {formatDate(person.first_seen)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Camera className="h-4 w-4" />
                      <span>Last: {formatDate(person.last_seen)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Users className="h-4 w-4" />
                      <span>{person.contact_count || 0} contacts</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Actions */}
              <div className="mt-4 flex items-center gap-2">
                <button 
                  onClick={() => handleViewContacts(person)}
                  className="flex-1 btn-secondary text-sm py-2"
                >
                  <Eye className="h-4 w-4 mr-1" />
                  View Contacts
                </button>
                <button 
                  onClick={() => handleLinkToPerson(person)}
                  className="flex-1 btn-primary text-sm py-2"
                >
                  <Link2 className="h-4 w-4 mr-1" />
                  Link
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Contact Details Modal */}
      {selectedPerson && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-100">
                    {selectedPerson.snapshot ? (
                      <img 
                        src={`data:image/jpeg;base64,${selectedPerson.snapshot}`}
                        alt={selectedPerson.temp_id}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <User className="h-8 w-8 text-gray-300" />
                      </div>
                    )}
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-800">
                      {selectedPerson.temp_id}
                    </h2>
                    <p className="text-gray-500">Contact History</p>
                  </div>
                </div>
                <button 
                  onClick={() => setSelectedPerson(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {loadingContacts ? (
                <div className="flex justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                </div>
              ) : contacts.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No contacts recorded</p>
              ) : (
                <div className="space-y-3">
                  {contacts.map((contact, idx) => (
                    <div 
                      key={idx}
                      className={`p-4 rounded-lg border ${
                        contact.other_is_mdr 
                          ? 'bg-danger-50 border-danger-200' 
                          : 'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${
                            contact.other_is_mdr ? 'bg-danger-100' : 'bg-gray-200'
                          }`}>
                            {contact.other_is_mdr ? (
                              <AlertTriangle className="h-5 w-5 text-danger-600" />
                            ) : (
                              <User className="h-5 w-5 text-gray-600" />
                            )}
                          </div>
                          <div>
                            <p className="font-medium text-gray-800">
                              {contact.other_person}
                              {contact.other_is_mdr && (
                                <span className="ml-2 px-2 py-0.5 bg-danger-100 text-danger-700 rounded text-xs">
                                  MDR Patient
                                </span>
                              )}
                            </p>
                            <p className="text-sm text-gray-500">
                              {formatDate(contact.contact_time || contact.timestamp)}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-semibold text-gray-800">
                            {Math.round(contact.duration_sec || 0)}s
                          </p>
                          <p className="text-sm text-gray-500">Duration</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Link to Person Modal */}
      {showLinkModal && linkTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full">
            <div className="p-6 border-b">
              <h2 className="text-xl font-bold text-gray-800">Link to Registered Person</h2>
              <p className="text-gray-500">
                Link <strong>{linkTarget.temp_id}</strong> to a registered person
              </p>
            </div>
            
            <div className="p-6">
              <div className="relative mb-4">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search person..."
                  className="input pl-10"
                />
              </div>
              
              <div className="max-h-64 overflow-y-auto space-y-2">
                {filteredPersons.map((person) => (
                  <button
                    key={person.id || person.name}
                    onClick={() => handleConfirmLink(person.name)}
                    className="w-full p-3 rounded-lg border border-gray-200 hover:bg-primary-50 hover:border-primary-300 transition-colors text-left flex items-center gap-3"
                  >
                    <div className="p-2 bg-gray-100 rounded-lg">
                      <User className="h-5 w-5 text-gray-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-800">{person.name}</p>
                      <p className="text-sm text-gray-500">{person.role || 'Unknown Role'}</p>
                    </div>
                  </button>
                ))}
                {filteredPersons.length === 0 && (
                  <p className="text-gray-500 text-center py-4">No matching persons found</p>
                )}
              </div>
            </div>
            
            <div className="p-6 border-t bg-gray-50">
              <button
                onClick={() => {
                  setShowLinkModal(false);
                  setLinkTarget(null);
                  setSearchTerm('');
                }}
                className="w-full btn-secondary"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
