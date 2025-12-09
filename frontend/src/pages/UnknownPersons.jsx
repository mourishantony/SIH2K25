import { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { unknownAPI, personsAPI } from '../api';
import { 
  User, UserX, Camera, Clock, AlertTriangle, 
  RefreshCw, Eye, Search, Users, Shield, Link2,
  Trash2, UserPlus, X, Upload, Image, Phone, MapPin,
  Video, VideoOff
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
  const [filter, setFilter] = useState('all'); 
  

  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [registerTarget, setRegisterTarget] = useState(null);
  const [registerName, setRegisterName] = useState('');
  const [registerRole, setRegisterRole] = useState('patient');
  const [registerPhone, setRegisterPhone] = useState('');
  const [registerPlace, setRegisterPlace] = useState('');
  const [registerNotes, setRegisterNotes] = useState('');
  const [registerImages, setRegisterImages] = useState([]);
  const [maxImages, setMaxImages] = useState(50);
  const [registering, setRegistering] = useState(false);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);
  const [useWebcam, setUseWebcam] = useState(false);
  
  // Delete confirmation state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch settings on mount
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const res = await unknownAPI.getSettings();
        setMaxImages(res.data.max_images || 50);
      } catch (error) {
        console.error('Failed to fetch settings');
      }
    };
    fetchSettings();
  }, []);

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

  // Delete handlers
  const handleDeleteClick = (person) => {
    setDeleteTarget(person);
    setShowDeleteConfirm(true);
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    
    try {
      setDeleting(true);
      await unknownAPI.delete(deleteTarget.temp_id);
      toast.success(`Deleted ${deleteTarget.temp_id}`);
      setShowDeleteConfirm(false);
      setDeleteTarget(null);
      fetchUnknowns();
    } catch (error) {
      toast.error('Failed to delete person');
    } finally {
      setDeleting(false);
    }
  };

  // Register handlers
  const handleRegisterClick = (person) => {
    setRegisterTarget(person);
    setRegisterName('');
    setRegisterRole('patient');
    setRegisterPhone('');
    setRegisterPlace('');
    setRegisterNotes('');
    setRegisterImages([]);
    setUseWebcam(false);
    setShowRegisterModal(true);
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    const remainingSlots = maxImages - registerImages.length - 1; // -1 for the captured snapshot
    
    if (files.length > remainingSlots) {
      toast.error(`Can only add ${remainingSlots} more images (max ${maxImages} total)`);
      files.splice(remainingSlots);
    }
    
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target.result.split(',')[1]; // Remove data:image/...;base64, prefix
        setRegisterImages(prev => [...prev, base64]);
      };
      reader.readAsDataURL(file);
    });
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRemoveImage = (index) => {
    setRegisterImages(prev => prev.filter((_, i) => i !== index));
  };

  const captureFromWebcam = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const remainingSlots = maxImages - registerImages.length - 1;
        if (remainingSlots > 0) {
          // Extract base64 without the data:image/jpeg;base64, prefix
          const base64 = imageSrc.split(',')[1];
          setRegisterImages(prev => [...prev, base64]);
          toast.success('Image captured!');
        } else {
          toast.error(`Maximum ${maxImages} images reached`);
        }
      }
    }
  }, [maxImages, registerImages.length]);

  // Handle key press for webcam capture
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (showRegisterModal && useWebcam && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        captureFromWebcam();
      }
    };
    
    if (showRegisterModal && useWebcam) {
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, [showRegisterModal, useWebcam, captureFromWebcam]);

  const handleConfirmRegister = async () => {
    if (!registerTarget || !registerName.trim()) {
      toast.error('Please enter a name');
      return;
    }
    
    try {
      setRegistering(true);
      const res = await unknownAPI.register(registerTarget.temp_id, {
        name: registerName.trim(),
        role: registerRole,
        phone: registerPhone.trim(),
        place: registerPlace.trim(),
        notes: registerNotes.trim(),
        additional_images: registerImages,
      });
      toast.success(`Registered ${registerTarget.temp_id} as ${registerName}`);
      setShowRegisterModal(false);
      setRegisterTarget(null);
      fetchUnknowns();
    } catch (error) {
      const msg = error.response?.data?.detail || 'Failed to register person';
      toast.error(msg);
    } finally {
      setRegistering(false);
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
              <div className="mt-4 space-y-2">
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => handleViewContacts(person)}
                    className="flex-1 btn-secondary text-sm py-2"
                  >
                    <Eye className="h-4 w-4 mr-1" />
                    Contacts
                  </button>
                  <button 
                    onClick={() => handleRegisterClick(person)}
                    className="flex-1 btn-primary text-sm py-2"
                  >
                    <UserPlus className="h-4 w-4 mr-1" />
                    Register
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => handleLinkToPerson(person)}
                    className="flex-1 btn-secondary text-sm py-2"
                  >
                    <Link2 className="h-4 w-4 mr-1" />
                    Link
                  </button>
                  <button 
                    onClick={() => handleDeleteClick(person)}
                    className="flex-1 btn-secondary text-sm py-2 text-danger-600 hover:bg-danger-50"
                  >
                    <Trash2 className="h-4 w-4 mr-1" />
                    Delete
                  </button>
                </div>
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
                        <div className="text-right flex items-center gap-4">
                          <div>
                            <p className={`text-lg font-semibold ${
                              (contact.risk_percent || 0) >= 40 
                                ? 'text-danger-600' 
                                : (contact.risk_percent || 0) >= 20 
                                  ? 'text-warning-600' 
                                  : 'text-gray-800'
                            }`}>
                              {(contact.risk_percent || 0).toFixed(1)}%
                            </p>
                            <p className="text-sm text-gray-500">Risk</p>
                          </div>
                          <div>
                            <p className="text-lg font-semibold text-gray-800">
                              {Math.round(contact.duration_sec || contact.duration_seconds || 0)}s
                            </p>
                            <p className="text-sm text-gray-500">Duration</p>
                          </div>
                          <div>
                            <p className="text-lg font-semibold text-gray-800">
                              {contact.min_distance_meters !== undefined && contact.min_distance_meters !== null
                                ? `${contact.min_distance_meters.toFixed(2)}m`
                                : contact.distance_meters !== undefined && contact.distance_meters !== null
                                  ? `${contact.distance_meters.toFixed(2)}m`
                                  : 'N/A'}
                            </p>
                            <p className="text-sm text-gray-500">Distance</p>
                          </div>
                        </div>
                      </div>
                      {/* Risk progress bar */}
                      <div className="mt-2">
                        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all ${
                              (contact.risk_percent || 0) >= 40 
                                ? 'bg-danger-500' 
                                : (contact.risk_percent || 0) >= 20 
                                  ? 'bg-warning-500' 
                                  : 'bg-primary-500'
                            }`}
                            style={{ width: `${Math.min(contact.risk_percent || 0, 100)}%` }}
                          />
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

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && deleteTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full">
            <div className="p-6">
              <div className="flex items-center gap-4 mb-4">
                <div className="p-3 bg-danger-100 rounded-full">
                  <Trash2 className="h-6 w-6 text-danger-600" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-800">Delete Unknown Person</h2>
                  <p className="text-gray-500">This action cannot be undone</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg mb-4">
                <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-100">
                  {deleteTarget.snapshot ? (
                    <img 
                      src={`data:image/jpeg;base64,${deleteTarget.snapshot}`}
                      alt={deleteTarget.temp_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <User className="h-8 w-8 text-gray-300" />
                    </div>
                  )}
                </div>
                <div>
                  <p className="font-semibold text-gray-800">{deleteTarget.temp_id}</p>
                  <p className="text-sm text-gray-500">{deleteTarget.contact_count || 0} contacts will also be deleted</p>
                </div>
              </div>
              
              <p className="text-gray-600 mb-6">
                Are you sure you want to delete <strong>{deleteTarget.temp_id}</strong>? 
                All contact history for this person will be permanently removed.
              </p>
              
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setShowDeleteConfirm(false);
                    setDeleteTarget(null);
                  }}
                  className="flex-1 btn-secondary"
                  disabled={deleting}
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmDelete}
                  className="flex-1 bg-danger-600 hover:bg-danger-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                  disabled={deleting}
                >
                  {deleting ? (
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  ) : (
                    <>
                      <Trash2 className="h-4 w-4" />
                      Delete
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Register Modal */}
      {showRegisterModal && registerTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-xl shadow-xl max-w-lg w-full my-4">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary-100 rounded-lg">
                    <UserPlus className="h-6 w-6 text-primary-600" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-800">Register Person</h2>
                    <p className="text-gray-500">Convert {registerTarget.temp_id} to registered</p>
                  </div>
                </div>
                <button 
                  onClick={() => {
                    setShowRegisterModal(false);
                    setRegisterTarget(null);
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>
            
            <div className="p-6 max-h-[70vh] overflow-y-auto">
              {/* Preview */}
              <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg mb-6">
                <div className="w-20 h-20 rounded-lg overflow-hidden bg-gray-100">
                  {registerTarget.snapshot ? (
                    <img 
                      src={`data:image/jpeg;base64,${registerTarget.snapshot}`}
                      alt={registerTarget.temp_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <User className="h-10 w-10 text-gray-300" />
                    </div>
                  )}
                </div>
                <div>
                  <p className="font-semibold text-gray-800">{registerTarget.temp_id}</p>
                  <p className="text-sm text-gray-500">
                    First seen: {formatDate(registerTarget.first_seen)}
                  </p>
                  <p className="text-sm text-gray-500">
                    {registerTarget.contact_count || 0} contacts recorded
                  </p>
                </div>
              </div>
              
              {/* Form */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Name <span className="text-danger-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={registerName}
                    onChange={(e) => setRegisterName(e.target.value)}
                    placeholder="Enter person's name"
                    className="input"
                    autoFocus
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Role
                  </label>
                  <select
                    value={registerRole}
                    onChange={(e) => setRegisterRole(e.target.value)}
                    className="input"
                  >
                    <option value="patient">Patient (P###)</option>
                    <option value="doctor">Doctor (D###)</option>
                    <option value="visitor">Visitor (V###)</option>
                    <option value="nurse">Nurse (N###)</option>
                    <option value="worker">Worker (W###)</option>
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      <div className="flex items-center gap-1">
                        <Phone className="h-4 w-4" />
                        Phone
                      </div>
                    </label>
                    <input
                      type="tel"
                      value={registerPhone}
                      onChange={(e) => setRegisterPhone(e.target.value)}
                      placeholder="Phone number"
                      className="input"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      <div className="flex items-center gap-1">
                        <MapPin className="h-4 w-4" />
                        Place
                      </div>
                    </label>
                    <input
                      type="text"
                      value={registerPlace}
                      onChange={(e) => setRegisterPlace(e.target.value)}
                      placeholder="Location/Ward"
                      className="input"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Notes
                  </label>
                  <textarea
                    value={registerNotes}
                    onChange={(e) => setRegisterNotes(e.target.value)}
                    placeholder="Optional notes about this person"
                    className="input min-h-[80px]"
                    rows={3}
                  />
                </div>
                
                {/* Additional Images - Webcam or Upload */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1">
                        <Image className="h-4 w-4" />
                        Additional Face Images
                      </div>
                      <span className="text-xs text-gray-500">
                        {registerImages.length + 1}/{maxImages} images
                      </span>
                    </div>
                  </label>
                  
                  {/* Toggle between Webcam and Upload */}
                  <div className="flex items-center gap-2 mb-3">
                    <button
                      type="button"
                      onClick={() => setUseWebcam(false)}
                      className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
                        !useWebcam 
                          ? 'bg-primary-100 text-primary-700 border-2 border-primary-500' 
                          : 'bg-gray-100 text-gray-600 border-2 border-transparent hover:bg-gray-200'
                      }`}
                    >
                      <Upload className="h-4 w-4" />
                      Upload
                    </button>
                    <button
                      type="button"
                      onClick={() => setUseWebcam(true)}
                      className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
                        useWebcam 
                          ? 'bg-primary-100 text-primary-700 border-2 border-primary-500' 
                          : 'bg-gray-100 text-gray-600 border-2 border-transparent hover:bg-gray-200'
                      }`}
                    >
                      <Video className="h-4 w-4" />
                      Webcam
                    </button>
                  </div>
                  
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-4">
                    {/* Webcam Mode */}
                    {useWebcam ? (
                      <div className="space-y-3">
                        <div 
                          className="relative rounded-lg overflow-hidden bg-black cursor-pointer"
                          onClick={captureFromWebcam}
                        >
                          <Webcam
                            ref={webcamRef}
                            audio={false}
                            screenshotFormat="image/jpeg"
                            className="w-full"
                            videoConstraints={{
                              width: 320,
                              height: 240,
                              facingMode: "user"
                            }}
                          />
                          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className="w-24 h-32 border-2 border-dashed border-white/50 rounded-lg" />
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={captureFromWebcam}
                            className="flex-1 btn-primary text-sm flex items-center justify-center gap-2"
                            disabled={registerImages.length + 1 >= maxImages}
                          >
                            <Camera className="h-4 w-4" />
                            Capture (Enter/Space)
                          </button>
                          <button
                            type="button"
                            onClick={() => setRegisterImages([])}
                            className="btn-secondary text-sm px-3"
                            title="Clear all captured"
                          >
                            <RefreshCw className="h-4 w-4" />
                          </button>
                        </div>
                        <p className="text-xs text-gray-500 text-center">
                          Click video or press Enter/Space to capture
                        </p>
                      </div>
                    ) : (
                      /* Upload Mode */
                      <>
                        <div className="flex flex-wrap gap-2 mb-3">
                          {/* Show the original snapshot as first image */}
                          {registerTarget.snapshot && (
                            <div className="relative group">
                              <img 
                                src={`data:image/jpeg;base64,${registerTarget.snapshot}`}
                                alt="Original"
                                className="w-16 h-16 rounded-lg object-cover border-2 border-primary-500"
                              />
                              <span className="absolute -top-2 -right-2 bg-primary-500 text-white text-xs px-1 rounded">
                                1
                              </span>
                            </div>
                          )}
                          
                          {/* Show uploaded additional images */}
                          {registerImages.map((img, idx) => (
                            <div key={idx} className="relative group">
                              <img 
                                src={`data:image/jpeg;base64,${img}`}
                                alt={`Additional ${idx + 2}`}
                                className="w-16 h-16 rounded-lg object-cover border border-gray-300"
                              />
                              <button
                                type="button"
                                onClick={() => handleRemoveImage(idx)}
                                className="absolute -top-2 -right-2 bg-danger-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <X className="h-3 w-3" />
                              </button>
                              <span className="absolute -bottom-1 -right-1 bg-gray-600 text-white text-xs px-1 rounded">
                                {idx + 2}
                              </span>
                            </div>
                          ))}
                        </div>
                        
                        {registerImages.length + 1 < maxImages && (
                          <div className="flex items-center gap-2">
                            <input
                              type="file"
                              ref={fileInputRef}
                              onChange={handleImageUpload}
                              accept="image/*"
                              multiple
                              className="hidden"
                            />
                            <button
                              type="button"
                              onClick={() => fileInputRef.current?.click()}
                              className="btn-secondary text-sm flex items-center gap-2"
                            >
                              <Upload className="h-4 w-4" />
                              Upload Images
                            </button>
                            <span className="text-xs text-gray-500">
                              Max {maxImages - registerImages.length - 1} more
                            </span>
                          </div>
                        )}
                      </>
                    )}
                    
                    {/* Show captured/uploaded images when using webcam */}
                    {useWebcam && registerImages.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs text-gray-500 mb-2">Captured ({registerImages.length} images):</p>
                        <div className="flex flex-wrap gap-2">
                          {registerTarget.snapshot && (
                            <div className="relative">
                              <img 
                                src={`data:image/jpeg;base64,${registerTarget.snapshot}`}
                                alt="Original"
                                className="w-12 h-12 rounded object-cover border-2 border-primary-500"
                              />
                              <span className="absolute -top-1 -right-1 bg-primary-500 text-white text-xs px-1 rounded">1</span>
                            </div>
                          )}
                          {registerImages.map((img, idx) => (
                            <div key={idx} className="relative group">
                              <img 
                                src={`data:image/jpeg;base64,${img}`}
                                alt={`Capture ${idx + 2}`}
                                className="w-12 h-12 rounded object-cover border border-gray-300"
                              />
                              <button
                                type="button"
                                onClick={() => handleRemoveImage(idx)}
                                className="absolute -top-1 -right-1 bg-danger-500 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <X className="h-3 w-3" />
                              </button>
                              <span className="absolute -bottom-0.5 -right-0.5 bg-gray-600 text-white text-xs px-1 rounded">{idx + 2}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Adding more face images improves recognition accuracy.
                  </p>
                </div>
              </div>
              
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-700">
                  <strong>Note:</strong> The face images will be used for future recognition. 
                  Contact history will be preserved and linked to the new profile.
                  The unknown person record will be removed after registration.
                </p>
              </div>
            </div>
            
            <div className="p-6 border-t bg-gray-50 flex gap-3">
              <button
                onClick={() => {
                  setShowRegisterModal(false);
                  setRegisterTarget(null);
                }}
                className="flex-1 btn-secondary"
                disabled={registering}
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmRegister}
                className="flex-1 btn-primary flex items-center justify-center gap-2"
                disabled={registering || !registerName.trim()}
              >
                {registering ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                ) : (
                  <>
                    <UserPlus className="h-4 w-4" />
                    Register Person
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
