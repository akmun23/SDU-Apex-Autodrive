using System;
using System.Collections;
using SocketIO;
using UnityEngine;

// Development-source-player only: mirrors the GUI connection toggle after
// scene Start handlers have registered, then starts Socket.IO explicitly.
public sealed class BatchModeSocketBootstrap : MonoBehaviour
{
    [SerializeField] private SocketIOComponent socketIO;
    [SerializeField] private SocketConnection connection;

    private IEnumerator Start()
    {
        Debug.Log($"[BatchModeSocketBootstrap] Start batch={Application.isBatchMode}");
        if (!Application.isBatchMode || string.IsNullOrEmpty(
                Environment.GetEnvironmentVariable("SDU_APEX_DYNAMICS_DIR")))
            yield break;
        if (socketIO == null)
            throw new InvalidOperationException("Socket.IO reference is missing");

        socketIO.autoConnect = false;
        Debug.Log("[BatchModeSocketBootstrap] delayed socket activation armed");
        yield return null;

        if (connection == null)
            throw new InvalidOperationException("GUI socket toggle is missing");
        connection.ToggleSocketConnection();
        Debug.Log("[BatchModeSocketBootstrap] GUI socket toggle invoked");
        yield return null;

        socketIO.Connect();
        Debug.Log("[BatchModeSocketBootstrap] Socket.IO Connect invoked");
    }
}
