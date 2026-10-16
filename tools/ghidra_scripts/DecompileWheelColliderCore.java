// Decompile Unity's symbol-bearing WheelCollider integration methods.
// Import UnityPlayer_s.debug with -noanalysis and run as a post-script.
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.listing.Function;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;

public class DecompileWheelColliderCore extends GhidraScript {
    private static final String[][] TARGETS = {
        {"0x2343130", "555", "VehicleComputeTireForce"},
        {"0x2341b40", "229", "WheelCollider_SetForwardFriction"},
        {"0x2341c30", "229", "WheelCollider_SetSidewaysFriction"},
        {"0x2342870", "806", "WheelCollider_AddWheelToExistingVehicle"},
        {"0x2342ba0", "1412", "WheelCollider_AddWheelToRecreatedVehicle"},
        {"0x23420f0", "898", "WheelCollider_UpdateVehicleSpringCMOffsetsAndMasses"},
        {"0x23446b0", "306", "WheelCollider_ConfigureVehicleSubsteps"},
        {"0x2344dc0", "520", "VehiclesManager_AddVehicle"},
        {"0x2345750", "626", "VehicleConfig_AddVehicle"}
    };

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: DecompileWheelColliderCore.java <output-path>");
        }
        PrintWriter report = new PrintWriter(new BufferedWriter(new FileWriter(args[0])));
        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);
        try {
            for (String[] target : TARGETS) {
                Address entry = currentProgram.getAddressFactory()
                        .getDefaultAddressSpace().getAddress(target[0]);
                Address end = entry.add(Long.parseLong(target[1]) - 1);
                emit(report, "\n===== " + target[2] + " @ " + entry
                        + " size=" + target[1] + " =====");
                disassemble(entry);
                Function function = getFunctionAt(entry);
                if (function == null) {
                    function = createFunction(entry, target[2]);
                }
                if (function == null) {
                    emit(report, "FAILED: no function created");
                    continue;
                }
                if (function.getBody().getNumAddresses() != Long.parseLong(target[1])) {
                    try {
                        function.setBody(new AddressSet(entry, end));
                    } catch (Exception e) {
                        emit(report, "NOTE: retaining imported function body: "
                                + e.getClass().getSimpleName() + ": " + e.getMessage());
                    }
                }
                DecompileResults result = decompiler.decompileFunction(function, 120, monitor);
                if (result != null && result.decompileCompleted()
                        && result.getDecompiledFunction() != null) {
                    emit(report, result.getDecompiledFunction().getC());
                } else {
                    emit(report, "DECOMPILE FAILED: "
                            + (result == null ? "null result" : result.getErrorMessage()));
                }
            }
        } finally {
            decompiler.dispose();
            report.close();
        }
    }

    private void emit(PrintWriter report, String line) {
        println(line);
        report.println(line);
    }
}
