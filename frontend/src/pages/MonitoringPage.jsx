import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Camera, Video, Upload, Play, Square, Settings, Trash2, 
  RefreshCw, Wifi, WifiOff, AlertCircle, CheckCircle, Loader2,
  Monitor, Webcam
} from 'lucide-react';
import { monitoringAPI } from '../api';
import toast from 'react-hot-toast';

const INPUT_MODES = [
  { id: 'video', label: 'Video File', icon: Video },
  { id: 'webcam', label: 'Live Webcam', icon: Webcam },
];

export default function MonitoringPage() {
  
  const [inputMode, setInputMode] = useState('video');
  const [videos, setVideos] = useState([]);
  const [selectedFrontVideo, setSelectedFrontVideo] = useState(null);
  const [selectedSideVideo, setSelectedSideVideo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadType, setUploadType] = useState('front');
  
  
  const [webcams, setWebcams] = useState([]);
  const [frontCamIndex, setFrontCamIndex] = useState(0);
  const [sideCamIndex, setSideCamIndex] = useState(1);
  
 
  const [isRunning, setIsRunning] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('idle');
  
 
  const [frameData, setFrameData] = useState({});
  const [combinedFrame, setCombinedFrame] = useState(null);
  const [detectedPersons, setDetectedPersons] = useState({ front: [], side: [] });
  const [activeContacts, setActiveContacts] = useState([]);
  const [monitorStats, setMonitorStats] = useState({});
  
  
  const [config, setConfig] = useState({
    use_gpu: false,
    min_confidence: 0.35,
    threshold: 0.32,
    base_rate: 0.02,
    event_penalty: 0.05,
  });
  const [showConfig, setShowConfig] = useState(false);
  
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);

 
  useEffect(() => {
    fetchVideos();
    checkStatus();
    
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

 
  useEffect(() => {
    if (inputMode === 'webcam' && webcams.length === 0) {
      enumerateWebcams();
    }
  }, [inputMode]);


  useEffect(() => {
    const frontVids = videos.filter(v => v.video_type === 'front');
    const sideVids = videos.filter(v => v.video_type === 'side');
    

    if (frontVids.length === 1 && !selectedFrontVideo) {
      setSelectedFrontVideo(frontVids[0].filename);
    }
    if (sideVids.length === 1 && !selectedSideVideo) {
      setSelectedSideVideo(sideVids[0].filename);
    }
  }, [videos]);

  const fetchVideos = async () => {
    try {
      const response = await monitoringAPI.listVideos();
      setVideos(response.data.videos || []);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  const checkStatus = async () => {
    try {
      const response = await monitoringAPI.getStatus();
      const data = response.data;
      
      setStatus(data.status);
      setSessionId(data.session_id);
      setIsRunning(data.status === 'running' || data.status === 'starting');
      
      if (data.status === 'running') {
       
        connectWebSocket();
      }
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  const enumerateWebcams = async () => {
    try {
     
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      
      
      stream.getTracks().forEach(track => track.stop());
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      setWebcams(videoDevices);
    } catch (error) {
      console.error('Error enumerating webcams:', error);
      toast.error('Could not access webcams. Please grant camera permissions.');
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    
    if (!file.type.startsWith('video/')) {
      toast.error('Please select a video file');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      await monitoringAPI.uploadVideo(file, uploadType, (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(progress);
      });
      
      toast.success(`${uploadType} video uploaded successfully`);
      fetchVideos();
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(error.response?.data?.detail || 'Failed to upload video');
    } finally {
      setUploading(false);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDeleteVideo = async (filename) => {
    if (!confirm(`Delete video "${filename}"?`)) return;

    try {
      await monitoringAPI.deleteVideo(filename);
      toast.success('Video deleted');
      fetchVideos();
      
    
      if (selectedFrontVideo === filename) setSelectedFrontVideo(null);
      if (selectedSideVideo === filename) setSelectedSideVideo(null);
    } catch (error) {
      toast.error('Failed to delete video');
    }
  };

  const startMonitoring = async () => {
    try {

      let monitorConfig;
      
      if (inputMode === 'video') {
        if (!selectedFrontVideo || !selectedSideVideo) {
          toast.error('Please select both front and side videos');
          return;
        }
        
       
        const frontVideo = videos.find(v => v.filename === selectedFrontVideo);
        const sideVideo = videos.find(v => v.filename === selectedSideVideo);
        
        if (!frontVideo?.path || !sideVideo?.path) {
          toast.error('Video paths not found. Please re-upload the videos.');
          return;
        }
        
        console.log('Starting with videos:', { front: frontVideo.path, side: sideVideo.path });
        
        monitorConfig = {
          mode: 'video',
          front_video_path: frontVideo.path,
          side_video_path: sideVideo.path,
          ...config,
        };
      } else {
        if (webcams.length === 0) {
          await enumerateWebcams();
        }
        
        monitorConfig = {
          mode: 'webcam',
          front_camera_index: frontCamIndex,
          side_camera_index: sideCamIndex,
          ...config,
        };
      }

      const response = await monitoringAPI.startMonitoring(monitorConfig);
      
      setSessionId(response.data.session_id);
      setIsRunning(true);
      setStatus('starting');
      
      
      connectWebSocket();
      
      toast.success('Monitoring started');
    } catch (error) {
      console.error('Start error:', error);
      toast.error(error.response?.data?.detail || 'Failed to start monitoring');
    }
  };

  const stopMonitoring = async () => {
    try {
      await monitoringAPI.stopMonitoring();
      
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      setIsRunning(false);
      setConnected(false);
      setSessionId(null);
      setStatus('idle');
      setFrameData({});
      setCombinedFrame(null);
      setDetectedPersons({ front: [], side: [] });
      setActiveContacts([]);
      setMonitorStats({});
      
      toast.success('Monitoring stopped');
    } catch (error) {
      toast.error('Failed to stop monitoring');
    }
  };

  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const wsUrl = monitoringAPI.getWebSocketUrl();
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'frame') {
     
          if (data.frame_base64) {
            setCombinedFrame(data.frame_base64);
          }
          
          if (data.view && data.frame) {
            setFrameData(prev => ({
              ...prev,
              [data.view]: data.frame,
            }));
          }
    
          if (data.detected_persons) {
            setDetectedPersons(data.detected_persons);
          }
          if (data.active_contacts) {
            setActiveContacts(data.active_contacts);
          }
          if (data.stats) {
            setMonitorStats(data.stats);
          }
        } else if (data.type === 'collision') {
          
          toast.custom((t) => (
            <div className={`${t.visible ? 'animate-fadeIn' : 'opacity-0'} bg-red-600 text-white p-4 rounded-lg shadow-lg max-w-md`}>
              <div className="flex items-start gap-3">
                <AlertCircle className="h-6 w-6 flex-shrink-0" />
                <div>
                  <p className="font-bold">🚨 Contact Detected!</p>
                  <p className="text-sm mt-1">
                    {data.person1} contacted {data.person2}
                  </p>
                </div>
              </div>
            </div>
          ), { duration: 5000 });
        } else if (data.type === 'status') {
          setStatus(data.status);
          if (data.status === 'idle') {
            setIsRunning(false);
          } else if (data.status === 'running') {
            setIsRunning(true);
          }
        } else if (data.type === 'error') {
          toast.error(data.message);
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };
  }, []);

  const canStart = () => {
    if (inputMode === 'video') {
      return selectedFrontVideo && selectedSideVideo;
    }
    return true; 
  };

  const frontVideos = videos.filter(v => v.video_type === 'front');
  const sideVideos = videos.filter(v => v.video_type === 'side');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">AI Monitoring</h1>
          <p className="text-gray-600">Real-time contact tracing with AI detection</p>
        </div>
        <div className="flex items-center gap-3">
          {isRunning && (
            <>
              {connected ? (
                <span className="flex items-center gap-2 text-green-600">
                  <Wifi className="h-5 w-5" />
                  Connected
                </span>
              ) : (
                <span className="flex items-center gap-2 text-yellow-600">
                  <WifiOff className="h-5 w-5" />
                  Reconnecting...
                </span>
              )}
              <span className={`px-3 py-1 rounded-full text-sm ${
                status === 'running' ? 'bg-green-100 text-green-700' :
                status === 'starting' ? 'bg-yellow-100 text-yellow-700' :
                'bg-gray-100 text-gray-700'
              }`}>
                {status}
              </span>
            </>
          )}
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="btn btn-secondary"
          >
            <Settings className="h-4 w-4" />
            Config
          </button>
        </div>
      </div>

      {/* Config Panel */}
      {showConfig && (
        <div className="card animate-fadeIn">
          <h3 className="text-lg font-medium mb-4">Monitoring Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Confidence
              </label>
              <input
                type="number"
                step="0.05"
                min="0.1"
                max="1"
                value={config.min_confidence}
                onChange={(e) => setConfig(prev => ({ ...prev, min_confidence: parseFloat(e.target.value) }))}
                className="input-field"
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Recognition Threshold
              </label>
              <input
                type="number"
                step="0.05"
                min="0.1"
                max="1"
                value={config.threshold}
                onChange={(e) => setConfig(prev => ({ ...prev, threshold: parseFloat(e.target.value) }))}
                className="input-field"
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base Rate
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="0.5"
                value={config.base_rate}
                onChange={(e) => setConfig(prev => ({ ...prev, base_rate: parseFloat(e.target.value) }))}
                className="input-field"
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Event Penalty
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="0.5"
                value={config.event_penalty}
                onChange={(e) => setConfig(prev => ({ ...prev, event_penalty: parseFloat(e.target.value) }))}
                className="input-field"
                disabled={isRunning}
              />
            </div>
            <div className="flex items-center">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.use_gpu}
                  onChange={(e) => setConfig(prev => ({ ...prev, use_gpu: e.target.checked }))}
                  className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  disabled={isRunning}
                />
                <span className="text-sm font-medium text-gray-700">Use GPU Acceleration</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Input Mode Selection */}
      <div className="flex gap-4">
        {INPUT_MODES.map(mode => {
          const Icon = mode.icon;
          return (
            <button
              key={mode.id}
              onClick={() => !isRunning && setInputMode(mode.id)}
              disabled={isRunning}
              className={`flex-1 p-4 rounded-lg border-2 transition-all ${
                inputMode === mode.id 
                  ? 'border-primary-500 bg-primary-50' 
                  : 'border-gray-200 hover:border-gray-300'
              } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <Icon className={`h-8 w-8 mx-auto mb-2 ${inputMode === mode.id ? 'text-primary-600' : 'text-gray-400'}`} />
              <span className={`block font-medium ${inputMode === mode.id ? 'text-primary-700' : 'text-gray-600'}`}>
                {mode.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Input Configuration */}
      <div className="card">
        {inputMode === 'video' ? (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium">Video Input</h3>
              <div className="flex gap-2">
                <select
                  value={uploadType}
                  onChange={(e) => setUploadType(e.target.value)}
                  className="input-field w-auto"
                  disabled={uploading || isRunning}
                >
                  <option value="front">Front Camera</option>
                  <option value="side">Side Camera</option>
                </select>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading || isRunning}
                  className="btn btn-secondary"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {uploadProgress}%
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      Upload
                    </>
                  )}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>
            </div>

            {/* Video Selection */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Front Videos */}
              <div>
                <h4 className="font-medium text-gray-700 mb-2">Front Camera Videos</h4>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {frontVideos.length === 0 ? (
                    <p className="text-gray-500 text-sm py-4 text-center">No front videos uploaded</p>
                  ) : (
                    frontVideos.map((video) => (
                      <div
                        key={video.filename}
                        onClick={() => !isRunning && setSelectedFrontVideo(video.filename)}
                        className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all text-sm ${
                          selectedFrontVideo === video.filename
                            ? 'border-primary-500 bg-primary-50'
                            : 'border-gray-200 hover:border-gray-300'
                        } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <Video className="h-4 w-4 text-gray-400 flex-shrink-0" />
                          <span className="truncate">{video.filename}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          {selectedFrontVideo === video.filename && (
                            <CheckCircle className="h-4 w-4 text-primary-600" />
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteVideo(video.filename);
                            }}
                            disabled={isRunning}
                            className="p-1 text-gray-400 hover:text-red-500"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Side Videos */}
              <div>
                <h4 className="font-medium text-gray-700 mb-2">Side Camera Videos</h4>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {sideVideos.length === 0 ? (
                    <p className="text-gray-500 text-sm py-4 text-center">No side videos uploaded</p>
                  ) : (
                    sideVideos.map((video) => (
                      <div
                        key={video.filename}
                        onClick={() => !isRunning && setSelectedSideVideo(video.filename)}
                        className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all text-sm ${
                          selectedSideVideo === video.filename
                            ? 'border-primary-500 bg-primary-50'
                            : 'border-gray-200 hover:border-gray-300'
                        } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <Video className="h-4 w-4 text-gray-400 flex-shrink-0" />
                          <span className="truncate">{video.filename}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          {selectedSideVideo === video.filename && (
                            <CheckCircle className="h-4 w-4 text-primary-600" />
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteVideo(video.filename);
                            }}
                            disabled={isRunning}
                            className="p-1 text-gray-400 hover:text-red-500"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div>
            <h3 className="text-lg font-medium mb-4">Webcam Selection</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Front Camera (Camera Index)
                </label>
                <select
                  value={frontCamIndex}
                  onChange={(e) => setFrontCamIndex(parseInt(e.target.value))}
                  disabled={isRunning}
                  className="input-field"
                >
                  {webcams.length === 0 ? (
                    <option value="0">Camera 0 (Default)</option>
                  ) : (
                    webcams.map((cam, idx) => (
                      <option key={cam.deviceId} value={idx}>
                        {cam.label || `Camera ${idx}`}
                      </option>
                    ))
                  )}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Side Camera (Camera Index)
                </label>
                <select
                  value={sideCamIndex}
                  onChange={(e) => setSideCamIndex(parseInt(e.target.value))}
                  disabled={isRunning}
                  className="input-field"
                >
                  {webcams.length === 0 ? (
                    <option value="1">Camera 1 (Default)</option>
                  ) : (
                    webcams.map((cam, idx) => (
                      <option key={cam.deviceId} value={idx}>
                        {cam.label || `Camera ${idx}`}
                      </option>
                    ))
                  )}
                </select>
              </div>
            </div>
            <button
              onClick={enumerateWebcams}
              disabled={isRunning}
              className="mt-4 btn btn-secondary"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh Cameras
            </button>
            <p className="mt-2 text-sm text-gray-500">
              Note: Webcam mode requires cameras connected to the server machine.
            </p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        {!isRunning ? (
          <button
            onClick={startMonitoring}
            disabled={!canStart()}
            className="btn btn-primary px-8 py-3 text-lg"
          >
            <Play className="h-5 w-5" />
            Start Monitoring
          </button>
        ) : (
          <button
            onClick={stopMonitoring}
            className="btn bg-red-600 hover:bg-red-700 text-white px-8 py-3 text-lg"
          >
            <Square className="h-5 w-5" />
            Stop Monitoring
          </button>
        )}
      </div>

      {/* Live Preview */}
      {isRunning && (
        <div className="card animate-fadeIn">
          <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
            <Monitor className="h-5 w-5 text-primary-600" />
            Live Preview
            {combinedFrame && (
              <span className="ml-auto text-sm font-normal text-green-600 flex items-center gap-1">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                Live
              </span>
            )}
          </h3>
          
          {/* Combined Frame View */}
          {combinedFrame ? (
            <div className="relative bg-gray-900 rounded-lg overflow-hidden">
              <img
                src={`data:image/jpeg;base64,${combinedFrame}`}
                alt="Combined Camera View"
                className="w-full h-auto object-contain"
              />
              <div className="absolute top-2 left-2 bg-black/50 text-white px-2 py-1 rounded text-sm">
                Front View
              </div>
              <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-sm">
                Side View
              </div>
            </div>
          ) : (
            <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
              <div className="text-center text-gray-400">
                <Loader2 className="h-12 w-12 mx-auto mb-2 animate-spin" />
                <p>Initializing cameras...</p>
                <p className="text-sm mt-1">Loading AI models and processing first frame</p>
              </div>
            </div>
          )}
          
          {/* Stats Panel */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-primary-600">
                {detectedPersons.front?.length || 0}
              </p>
              <p className="text-sm text-gray-600">Front Persons</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-primary-600">
                {detectedPersons.side?.length || 0}
              </p>
              <p className="text-sm text-gray-600">Side Persons</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-orange-600">
                {monitorStats.total_contacts || 0}
              </p>
              <p className="text-sm text-gray-600">Total Contacts</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-red-600">
                {monitorStats.mdr_contacts || 0}
              </p>
              <p className="text-sm text-gray-600">MDR Contacts</p>
            </div>
          </div>
          
          {/* Detected Persons List */}
          {(detectedPersons.front?.length > 0 || detectedPersons.side?.length > 0) && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm font-medium text-gray-700 mb-2">Detected Persons:</p>
              <div className="flex flex-wrap gap-2">
                {[...new Set([...detectedPersons.front || [], ...detectedPersons.side || []])].map(person => (
                  <span 
                    key={person} 
                    className={`px-2 py-1 rounded text-sm ${
                      person.startsWith('Unknown_') 
                        ? 'bg-yellow-100 text-yellow-800' 
                        : 'bg-green-100 text-green-800'
                    }`}
                  >
                    {person}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Session Info */}
      {sessionId && (
        <div className="text-center text-sm text-gray-500">
          Session ID: {sessionId}
        </div>
      )}
    </div>
  );
}
