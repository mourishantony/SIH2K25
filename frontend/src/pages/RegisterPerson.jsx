import { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { personsAPI, faceAPI } from '../api';
import { Camera, Check, X, RefreshCw, User, Phone, MapPin, Save } from 'lucide-react';
import toast from 'react-hot-toast';

export default function RegisterPerson() {
  const [step, setStep] = useState(1); 
  const [personData, setPersonData] = useState({
    name: '',
    role: 'patient',
    phone: '',
    place: '',
    notes: ''
  });
  const [capturedImages, setCapturedImages] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [settings, setSettings] = useState({ total_samples: 50 });
  const [saving, setSaving] = useState(false);
  const [training, setTraining] = useState(false);
  const [createdPerson, setCreatedPerson] = useState(null);
  
  const webcamRef = useRef(null);

  useEffect(() => {
   
    faceAPI.getSettings().then(res => {
      setSettings(res.data);
    }).catch(err => {
      console.error('Error fetching settings:', err);
    });
  }, []);

  const handlePersonSubmit = async (e) => {
    e.preventDefault();
    
    if (!personData.name.trim()) {
      toast.error('Please enter a name');
      return;
    }

    setSaving(true);
    
    try {
      const response = await personsAPI.create(personData);
      setCreatedPerson(response.data);
      const personId = response.data.person_id || '';
      toast.success(`${personData.name} registered successfully! ID: ${personId}`);
      setStep(2);
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to register person';
      toast.error(message);
    } finally {
      setSaving(false);
    }
  };

  const captureImage = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCapturedImages(prev => [...prev, imageSrc]);
      }
    }
  }, [webcamRef]);

  const handleKeyPress = useCallback((e) => {
    if (step === 2 && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      captureImage();
    }
  }, [step, captureImage]);

  useEffect(() => {
    if (step === 2) {
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, [step, handleKeyPress]);

  const removeImage = (index) => {
    setCapturedImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmitFaces = async () => {
    if (capturedImages.length === 0) {
      toast.error('Please capture at least one image');
      return;
    }

    setTraining(true);
    
    try {
      const response = await faceAPI.uploadImages(
        createdPerson.name,
        capturedImages,
        true 
      );
      
      toast.success(`Training started! ${response.data.stored_count} images uploaded.`);
      
     
      setPersonData({ name: '', role: 'patient', phone: '', place: '', notes: '' });
      setCapturedImages([]);
      setCreatedPerson(null);
      setStep(1);
      
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to upload images';
      toast.error(message);
    } finally {
      setTraining(false);
    }
  };

  const skipFaceCapture = () => {
    toast.success('Person registered without face data');
    setPersonData({ name: '', role: 'patient', phone: '', place: '', notes: '' });
    setCapturedImages([]);
    setCreatedPerson(null);
    setStep(1);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fadeIn">
      <div>
        <h1 className="text-2xl font-bold text-gray-800">Register Person</h1>
        <p className="text-gray-500">Register a new patient, doctor, visitor, or worker</p>
      </div>

      {/* Progress Steps */}
      <div className="flex items-center gap-4">
        <div className={`flex items-center gap-2 ${step >= 1 ? 'text-primary-600' : 'text-gray-400'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? 'bg-primary-600 text-white' : 'bg-gray-200'}`}>
            1
          </div>
          <span className="font-medium">Person Details</span>
        </div>
        <div className="flex-1 h-1 bg-gray-200 rounded">
          <div className={`h-full bg-primary-600 rounded transition-all ${step >= 2 ? 'w-full' : 'w-0'}`} />
        </div>
        <div className={`flex items-center gap-2 ${step >= 2 ? 'text-primary-600' : 'text-gray-400'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-primary-600 text-white' : 'bg-gray-200'}`}>
            2
          </div>
          <span className="font-medium">Face Capture</span>
        </div>
      </div>

      {step === 1 ? (
        /* Step 1: Person Details Form */
        <div className="card">
          <form onSubmit={handlePersonSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <User className="inline h-4 w-4 mr-1" />
                  Full Name *
                </label>
                <input
                  type="text"
                  value={personData.name}
                  onChange={(e) => setPersonData({ ...personData, name: e.target.value })}
                  className="input"
                  placeholder="Enter full name"
                  disabled={saving}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Role *
                </label>
                <select
                  value={personData.role}
                  onChange={(e) => setPersonData({ ...personData, role: e.target.value })}
                  className="input"
                  disabled={saving}
                >
                  <option value="patient">Patient (P###)</option>
                  <option value="doctor">Doctor (D###)</option>
                  <option value="visitor">Visitor (V###)</option>
                  <option value="nurse">Nurse (N###)</option>
                  <option value="worker">Worker (W###)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Phone className="inline h-4 w-4 mr-1" />
                  Phone Number
                </label>
                <input
                  type="tel"
                  value={personData.phone}
                  onChange={(e) => setPersonData({ ...personData, phone: e.target.value })}
                  className="input"
                  placeholder="Enter phone number"
                  disabled={saving}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <MapPin className="inline h-4 w-4 mr-1" />
                  Place
                </label>
                <input
                  type="text"
                  value={personData.place}
                  onChange={(e) => setPersonData({ ...personData, place: e.target.value })}
                  className="input"
                  placeholder="Enter place/location"
                  disabled={saving}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Notes
              </label>
              <textarea
                value={personData.notes}
                onChange={(e) => setPersonData({ ...personData, notes: e.target.value })}
                className="input min-h-[100px]"
                placeholder="Additional notes..."
                disabled={saving}
              />
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={saving}
                className="btn-primary flex items-center gap-2"
              >
                {saving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    Saving...
                  </>
                ) : (
                  <>
                    Continue to Face Capture
                    <Camera className="h-4 w-4" />
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      ) : (
        /* Step 2: Face Capture */
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold">Capture Face Images</h2>
                <p className="text-sm text-gray-500">
                  For: <span className="font-medium text-gray-800">{createdPerson?.name}</span>
                  {createdPerson?.person_id && (
                    <span className="ml-2 px-2 py-0.5 bg-primary-100 text-primary-700 rounded text-xs font-bold">
                      {createdPerson.person_id}
                    </span>
                  )}
                </p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-primary-600">{capturedImages.length}</p>
                <p className="text-sm text-gray-500">/ {settings.total_samples} samples</p>
              </div>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
              <p className="text-sm text-yellow-800">
                <strong>Tips:</strong> Click the capture button, press Enter, or click on the video to capture. 
                Try different angles and with/without mask for better recognition.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Webcam */}
              <div>
                <div 
                  className="relative rounded-lg overflow-hidden bg-black cursor-pointer"
                  onClick={captureImage}
                >
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    className="w-full"
                    videoConstraints={{
                      width: 640,
                      height: 480,
                      facingMode: "user"
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="w-48 h-64 border-2 border-dashed border-white/50 rounded-lg" />
                  </div>
                </div>
                
                <div className="flex items-center gap-3 mt-4">
                  <button
                    onClick={captureImage}
                    className="flex-1 btn-primary flex items-center justify-center gap-2 py-3"
                  >
                    <Camera className="h-5 w-5" />
                    Capture (Enter)
                  </button>
                  <button
                    onClick={() => setCapturedImages([])}
                    className="btn-secondary px-4 py-3"
                    title="Clear all"
                  >
                    <RefreshCw className="h-5 w-5" />
                  </button>
                </div>
              </div>

              {/* Captured Images Grid */}
              <div>
                <p className="text-sm font-medium text-gray-700 mb-2">Captured Images</p>
                <div className="grid grid-cols-4 gap-2 max-h-80 overflow-y-auto p-1">
                  {capturedImages.map((img, index) => (
                    <div key={index} className="relative group">
                      <img
                        src={img}
                        alt={`Capture ${index + 1}`}
                        className="w-full h-16 object-cover rounded"
                      />
                      <button
                        onClick={() => removeImage(index)}
                        className="absolute -top-1 -right-1 bg-danger-500 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X className="h-3 w-3" />
                      </button>
                      <span className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs text-center">
                        {index + 1}
                      </span>
                    </div>
                  ))}
                  {capturedImages.length === 0 && (
                    <div className="col-span-4 text-center py-8 text-gray-400">
                      No images captured yet
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between">
            <button
              onClick={skipFaceCapture}
              className="btn-secondary"
            >
              Skip Face Capture
            </button>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => setStep(1)}
                className="btn-secondary"
              >
                Back
              </button>
              <button
                onClick={handleSubmitFaces}
                disabled={capturedImages.length === 0 || training}
                className="btn-success flex items-center gap-2"
              >
                {training ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    Training...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save & Train ({capturedImages.length} images)
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
