import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';

// Tools that can require confirmation before execution
const TOOLS = [
    { id: 'generate_cad', label: 'Generate CAD' },
    { id: 'iterate_cad', label: 'Iterate CAD' },
    { id: 'run_web_agent', label: 'Web Agent' },
    { id: 'write_file', label: 'Write File' },
    { id: 'read_directory', label: 'Read Directory' },
    { id: 'read_file', label: 'Read File' },
    { id: 'create_project', label: 'Create Project' },
    { id: 'switch_project', label: 'Switch Project' },
    { id: 'list_projects', label: 'List Projects' },
];

const SettingsWindow = ({
    socket,
    micDevices,
    speakerDevices,
    webcamDevices,
    selectedMicId,
    setSelectedMicId,
    selectedSpeakerId,
    setSelectedSpeakerId,
    selectedWebcamId,
    setSelectedWebcamId,
    cursorSensitivity,
    setCursorSensitivity,
    isCameraFlipped,
    setIsCameraFlipped,
    handleFileUpload,
    onClose
}) => {
    const [permissions, setPermissions] = useState({});
    const [voiceGender, setVoiceGender] = useState('male');
    const [userName, setUserName] = useState('');
    const [loadedUserName, setLoadedUserName] = useState('');
    const [personality, setPersonality] = useState({
        enabled: false,
        auto_project_naming: false,
        humor_in_logs: false,
        delight_moments: false,
        humor_rate: 0.06,
        delight_rate: 0.02,
        min_delight_interval_s: 600,
    });

    const saveUserName = () => {
        const next = String(userName || '').trim();
        if (!next) {
            socket.emit('forget_user_name');
            return;
        }
        socket.emit('set_identity', { identity: { user_name: next } });
    };

    const forgetUserName = () => {
        setUserName('');
        socket.emit('forget_user_name');
    };

    useEffect(() => {
        // Request initial settings
        socket.emit('get_settings');

        const handleSettings = (settings) => {
            console.log("Received settings:", settings);
            if (settings) {
                if (settings.tool_permissions) setPermissions(settings.tool_permissions);
                const ident = settings.identity || {};
                if (typeof ident.user_name === 'string') {
                    setLoadedUserName(ident.user_name);
                    setUserName(prev => (prev ? prev : ident.user_name));
                }

                try {
                    const p = settings.personality;
                    if (p && typeof p === 'object') {
                        setPersonality(prev => ({
                            ...prev,
                            ...p,
                        }));
                    }
                } catch (_) {
                }

                const tts = settings.tts || {};
                const vg = tts.voice_gender;
                if (vg === 'male' || vg === 'female') {
                    setVoiceGender(vg);
                } else {
                    const vi = String(tts.voice_vi || '').toLowerCase();
                    const en = String(tts.voice_en || tts.voice || '').toLowerCase();
                    if (vi.includes('namminh') || en.includes('guy') || en.includes('andrew') || en.includes('eric')) {
                        setVoiceGender('male');
                    } else {
                        setVoiceGender('female');
                    }
                }
            }
        };

        socket.on('settings', handleSettings);

        return () => {
            socket.off('settings', handleSettings);
        };
    }, [socket]);

    const togglePermission = (toolId) => {
        const currentVal = permissions[toolId] !== false;
        const nextVal = !currentVal;
        socket.emit('update_settings', { tool_permissions: { [toolId]: nextVal } });
    };

    const toggleCameraFlip = () => {
        const newVal = !isCameraFlipped;
        setIsCameraFlipped(newVal);
        socket.emit('update_settings', { camera_flipped: newVal });
    };

    const updatePersonality = (patch) => {
        const next = { ...(personality || {}), ...(patch || {}) };
        setPersonality(next);
        socket.emit('update_settings', { personality: patch || {} });
    };

    const updateVoiceGender = (gender) => {
        const next = gender === 'female' ? 'female' : 'male';
        setVoiceGender(next);

        const voice_en = next === 'female' ? 'en-US-AriaNeural' : 'en-US-GuyNeural';
        const voice_vi = next === 'female' ? 'vi-VN-HoaiMyNeural' : 'vi-VN-NamMinhNeural';

        socket.emit('update_settings', {
            tts: {
                voice_gender: next,
                voice_en,
                voice_vi,
                voice: voice_en,
                auto_detect: true,
            }
        });
    };

    return (
        <div className="absolute top-20 right-10 bg-black/90 border border-cyan-500/50 p-4 rounded-lg z-50 w-80 backdrop-blur-xl shadow-[0_0_30px_rgba(6,182,212,0.2)] max-h-[80vh] overflow-y-auto custom-scrollbar">
            <div className="flex justify-between items-center mb-4 border-b border-cyan-900/50 pb-2">
                <h2 className="text-cyan-400 font-bold text-sm uppercase tracking-wider">Settings</h2>
                <button onClick={onClose} className="text-cyan-600 hover:text-cyan-400">
                    <X size={16} />
                </button>
            </div>

            {/* Identity Section */}
            <div className="mb-6">
                <h3 className="text-cyan-400 font-bold mb-3 text-xs uppercase tracking-wider opacity-80">Identity</h3>
                <div className="text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                    <label className="text-cyan-100/60 text-[10px] uppercase mb-1 block">Remembered Name</label>
                    <input
                        value={userName}
                        onChange={(e) => setUserName(e.target.value)}
                        placeholder={loadedUserName || 'User'}
                        className="w-full bg-gray-800 border border-cyan-800/50 rounded p-1.5 text-xs text-cyan-100 focus:border-cyan-400 outline-none"
                    />
                    <div className="flex gap-2 mt-2">
                        <button
                            onClick={saveUserName}
                            className="flex-1 text-[10px] uppercase tracking-wider bg-cyan-900/60 hover:bg-cyan-800/70 text-cyan-100 rounded py-1.5 border border-cyan-700/30"
                        >
                            Save
                        </button>
                        <button
                            onClick={forgetUserName}
                            className="flex-1 text-[10px] uppercase tracking-wider bg-gray-800 hover:bg-gray-700 text-cyan-100 rounded py-1.5 border border-cyan-900/30"
                        >
                            Forget
                        </button>
                    </div>
                </div>
            </div>

            <div className="mb-6">
                <h3 className="text-cyan-400 font-bold mb-3 text-xs uppercase tracking-wider opacity-80">Personality</h3>
                <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                        <span className="text-cyan-100/80">Enable Personality Signals</span>
                        <button
                            onClick={() => updatePersonality({ enabled: !personality.enabled })}
                            className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${personality.enabled ? 'bg-cyan-500/80' : 'bg-gray-700'}`}
                        >
                            <div
                                className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${personality.enabled ? 'translate-x-4' : 'translate-x-0'}`}
                            />
                        </button>
                    </div>

                    <div className="flex items-center justify-between text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                        <span className="text-cyan-100/80">Auto Project Naming</span>
                        <button
                            onClick={() => updatePersonality({ auto_project_naming: !personality.auto_project_naming })}
                            className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${personality.auto_project_naming ? 'bg-cyan-500/80' : 'bg-gray-700'}`}
                        >
                            <div
                                className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${personality.auto_project_naming ? 'translate-x-4' : 'translate-x-0'}`}
                            />
                        </button>
                    </div>

                    <div className="flex items-center justify-between text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                        <span className="text-cyan-100/80">Humor In Logs</span>
                        <button
                            onClick={() => updatePersonality({ humor_in_logs: !personality.humor_in_logs })}
                            className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${personality.humor_in_logs ? 'bg-cyan-500/80' : 'bg-gray-700'}`}
                        >
                            <div
                                className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${personality.humor_in_logs ? 'translate-x-4' : 'translate-x-0'}`}
                            />
                        </button>
                    </div>

                    <div className="flex items-center justify-between text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                        <span className="text-cyan-100/80">Delight Moments</span>
                        <button
                            onClick={() => updatePersonality({ delight_moments: !personality.delight_moments })}
                            className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${personality.delight_moments ? 'bg-cyan-500/80' : 'bg-gray-700'}`}
                        >
                            <div
                                className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${personality.delight_moments ? 'translate-x-4' : 'translate-x-0'}`}
                            />
                        </button>
                    </div>
                </div>
            </div>

            {/* TTS Settings */}
            <div className="mb-6">
                <h3 className="text-cyan-400 font-bold mb-3 text-xs uppercase tracking-wider opacity-80">Text-to-Speech</h3>
                <div className="text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                    <label className="text-cyan-100/60 text-[10px] uppercase mb-1 block">Voice Profile</label>
                    <select
                        value={voiceGender}
                        onChange={(e) => updateVoiceGender(e.target.value)}
                        className="w-full bg-gray-800 border border-cyan-800/50 rounded p-1.5 text-xs text-cyan-100 focus:border-cyan-400 outline-none"
                    >
                        <option value="female">Female</option>
                        <option value="male">Male</option>
                    </select>
                </div>
            </div>

            {/* Microphone Section */}
            <div className="mb-4">
                <h3 className="text-cyan-400 font-bold mb-2 text-xs uppercase tracking-wider opacity-80">Microphone</h3>
                <select
                    value={selectedMicId}
                    onChange={(e) => setSelectedMicId(e.target.value)}
                    className="w-full bg-gray-900 border border-cyan-800 rounded p-2 text-xs text-cyan-100 focus:border-cyan-400 outline-none"
                >
                    {micDevices.map((device, i) => (
                        <option key={device.deviceId} value={device.deviceId}>
                            {device.label || `Microphone ${i + 1}`}
                        </option>
                    ))}
                </select>
            </div>

            {/* Speaker Section */}
            <div className="mb-4">
                <h3 className="text-cyan-400 font-bold mb-2 text-xs uppercase tracking-wider opacity-80">Speaker</h3>
                <select
                    value={selectedSpeakerId}
                    onChange={(e) => setSelectedSpeakerId(e.target.value)}
                    className="w-full bg-gray-900 border border-cyan-800 rounded p-2 text-xs text-cyan-100 focus:border-cyan-400 outline-none"
                >
                    {speakerDevices.map((device, i) => (
                        <option key={device.deviceId} value={device.deviceId}>
                            {device.label || `Speaker ${i + 1}`}
                        </option>
                    ))}
                </select>
            </div>

            {/* Webcam Section */}
            <div className="mb-6">
                <h3 className="text-cyan-400 font-bold mb-2 text-xs uppercase tracking-wider opacity-80">Webcam</h3>
                <select
                    value={selectedWebcamId}
                    onChange={(e) => setSelectedWebcamId(e.target.value)}
                    className="w-full bg-gray-900 border border-cyan-800 rounded p-2 text-xs text-cyan-100 focus:border-cyan-400 outline-none"
                >
                    {webcamDevices.map((device, i) => (
                        <option key={device.deviceId} value={device.deviceId}>
                            {device.label || `Camera ${i + 1}`}
                        </option>
                    ))}
                </select>
            </div>

            {/* Gesture Control Section */}
            <div className="mb-6">
                <h3 className="text-cyan-400 font-bold mb-3 text-xs uppercase tracking-wider opacity-80">Gesture Control</h3>
                <div className="space-y-3">
                    <div>
                        <div className="flex justify-between mb-1">
                            <span className="text-xs text-cyan-100/80">Cursor Sensitivity</span>
                    <span className="text-xs text-cyan-500">{cursorSensitivity}x</span>
                </div>
                <input
                    type="range"
                    min="1.0"
                    max="5.0"
                    step="0.1"
                    value={cursorSensitivity}
                    onChange={(e) => setCursorSensitivity(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer h-1 bg-gray-800 rounded-lg appearance-none"
                />
            </div>
                <div className="flex items-center justify-between text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                    <span className="text-cyan-100/80">Flip Camera Horizontal</span>
                    <button
                        onClick={toggleCameraFlip}
                        className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${isCameraFlipped ? 'bg-cyan-500/80' : 'bg-gray-700'}`}
                    >
                        <div
                            className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${isCameraFlipped ? 'translate-x-4' : 'translate-x-0'}`}
                        />
                    </button>
                    </div>
                </div>
            </div>

            {/* Tool Permissions Section */}
            <div className="mb-6">
                <h3 className="text-cyan-400 font-bold mb-3 text-xs uppercase tracking-wider opacity-80">Tool Confirmations</h3>
                <p className="text-[10px] text-cyan-500/60 mb-2">Toggle ON to require confirmation before tool execution</p>
                <div className="space-y-2 max-h-40 overflow-y-auto pr-2 custom-scrollbar">
                    {TOOLS.map(tool => {
                        const isRequired = permissions[tool.id] !== false;
                        return (
                            <div key={tool.id} className="flex items-center justify-between text-xs bg-gray-900/50 p-2 rounded border border-cyan-900/30">
                                <span className="text-cyan-100/80">{tool.label}</span>
                                <button
                                    onClick={() => togglePermission(tool.id)}
                                    className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${isRequired ? 'bg-cyan-500/80' : 'bg-gray-700'}`}
                                >
                                    <div
                                        className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${isRequired ? 'translate-x-4' : 'translate-x-0'}`}
                                    />
                                </button>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Memory Section */}
            <div>
                <h3 className="text-cyan-400 font-bold mb-2 text-xs uppercase tracking-wider opacity-80">Memory Data</h3>
                <div className="flex flex-col gap-2">
                    <label className="text-[10px] text-cyan-500/60 uppercase">Upload Memory Text</label>
                    <input
                        type="file"
                        accept=".txt"
                        onChange={handleFileUpload}
                        className="text-xs text-cyan-100 bg-gray-900 border border-cyan-800 rounded p-2 file:mr-2 file:py-1 file:px-2 file:rounded-full file:border-0 file:text-[10px] file:font-semibold file:bg-cyan-900 file:text-cyan-400 hover:file:bg-cyan-800 cursor-pointer"
                    />
                </div>
            </div>
        </div>
    );
};

export default SettingsWindow;
