const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    ipcRenderer: {
        // Renderer → Main (fire-and-forget)
        send: (channel, ...args) => ipcRenderer.send(channel, ...args),

        // Renderer → Main → Renderer (request / reply)
        invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),

        // Main → Renderer (push events)
        // Returns the internal wrapper so the caller can removeListener later.
        on: (channel, listener) => {
            const wrapper = (_event, ...args) => listener(...args);
            ipcRenderer.on(channel, wrapper);
            return wrapper;
        },
        removeListener: (channel, listenerWrapper) => {
            ipcRenderer.removeListener(channel, listenerWrapper);
        },
    },
});
