// Headless Ghidra script for the exact-build Unity/PhysX vehicle call path.
// It creates only the named target functions, avoiding whole-player analysis.
import ghidra.app.cmd.disassemble.DisassembleCommand;
import ghidra.app.cmd.function.CreateFunctionCmd;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;

import java.io.FileWriter;
import java.io.PrintWriter;

public class ExportWheelCollider extends GhidraScript {
  private static final long[][] TARGETS = {
    {0x015127D0L, 0}, // WheelCollider::SetForwardFriction
    {0x015128B0L, 0}, // WheelCollider::SetSidewaysFriction
    {0x01511F80L, 0}, // WheelCollider::SetSuspensionSpring
    {0x01512190L, 0}, // WheelCollider::SetSuspensionDistance
    {0x01512340L, 0}, // WheelCollider::SetForceAppPointDistance
    {0x015126C0L, 0}, // WheelCollider::SetWheelDampingRate
    {0x01512530L, 0}, // WheelCollider::SetMass
    {0x015134B0L, 0}, // WheelCollider::AddWheelToExistingVehicle
    {0x015130C0L, 0}, // WheelCollider::ComputeWheelCentreOfMassOffset
    {0x01512D40L, 0}, // WheelCollider::UpdateVehicleSpringCMOffsetsAndMasses
    {0x01515490L, 0}, // WheelCollider::ResetSprungMasses
    {0x015153D0L, 0}, // WheelCollider::SetSprungMass
    {0x01513F70L, 0}, // WheelCollider::SetSuspensionExpansionLimited
    {0x01513FA0L, 0}, // WheelCollider::GetSuspensionExpansionLimited
    {0x01514840L, 0}, // WheelCollider::SetMotorTorque
    {0x01514920L, 0}, // WheelCollider::SetBrakeTorque
    {0x01514A00L, 0}, // WheelCollider::SetSteerAngle
    {0x015151C0L, 0}, // WheelCollider::GetRpm
    {0x01513D40L, 0}, // VehicleComputeTireForce
    {0x01515240L, 0}, // WheelCollider::ConfigureVehicleSubsteps
    {0x01515DB0L, 0}, // VehiclesManager::UpdateVehicles
    {0x01516E10L, 0}, // VehiclesManager::VehicleConfig::UpdateVehicles
    {0x01516480L, 0}, // VehiclesManager::VehicleWheelRaycastPreFilter
    {0x01B28FD0L, 0}, // physx::processSuspTireWheels
    {0x01B32E20L, 0}, // physx::PxVehicleUpdate::updateNoDrive
    {0x01B36C90L, 0}, // physx::PxVehicleComputeSprungMasses
    {0x00D3B7C0L, 0}, // Rigidbody::UpdateMassDistribution
    {0x00D39A20L, 0}, // Rigidbody::SetCenterOfMass
    {0x00D3C7F0L, 0}, // Rigidbody::SetAutomaticInertiaTensor
    {0x01B7A8E0L, 0}, // PhysX updateMassAndInertia
    {0x01B7AB20L, 0}, // PhysX setMassAndUpdateInertia
    {0x01B7AD50L, 0}, // PhysX PxRigidBodyExt::setMassAndUpdateInertia
    {0x01B82F70L, 0}, // PhysX computeMassAndInertia
    {0x01B84BC0L, 0}, // PhysX computeMassAndDiagInertia
    {0x01B7A5D0L, 0}, // PhysX InertiaTensorComputer::translate
  };

  @Override
  public void run() throws Exception {
    String[] args = getScriptArgs();
    if (args.length != 1) {
      throw new IllegalArgumentException("usage: ExportWheelCollider OUTPUT_PATH");
    }

    DecompInterface decompiler = new DecompInterface();
    decompiler.openProgram(currentProgram);
    FunctionManager functions = currentProgram.getFunctionManager();
    try (PrintWriter writer = new PrintWriter(new FileWriter(args[0]))) {
      writer.println("Program: " + currentProgram.getName());
      writer.println("Image base: " + currentProgram.getImageBase());
      for (long[] target : TARGETS) {
        // ELF ET_DYN symbols are link-time offsets; Ghidra imports this PIE
        // image at 0x100000, so translate each nm/objdump address by the image
        // base before looking up or creating the function.
        Address entry = currentProgram.getImageBase().add(target[0]);
        Function function = functions.getFunctionAt(entry);
        if (function == null) {
          new DisassembleCommand(entry, null, true)
              .applyTo(currentProgram, monitor);
          CreateFunctionCmd create = new CreateFunctionCmd(entry);
          create.applyTo(currentProgram, monitor);
          function = functions.getFunctionAt(entry);
        }
        writer.println("\n\n===== " + entry + " =====");
        if (function == null) {
          writer.println("ERROR: could not establish function at entry");
          continue;
        }
        writer.println("Symbol: " + function.getName());
        DecompileResults result = decompiler.decompileFunction(
            function, 180, monitor);
        if (!result.decompileCompleted()) {
          writer.println("DECOMPILE ERROR: " + result.getErrorMessage());
          continue;
        }
        writer.println(result.getDecompiledFunction().getC());
      }
    } finally {
      decompiler.dispose();
    }
  }
}
