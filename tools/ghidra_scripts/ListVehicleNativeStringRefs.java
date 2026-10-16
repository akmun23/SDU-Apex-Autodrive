// Find native UnityPlayer functions referenced by WheelCollider/vehicle
// strings. Run post-analysis with Ghidra analyzeHeadless.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.DataIterator;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceIterator;

public class ListVehicleNativeStringRefs extends GhidraScript {
    private static final String[] NEEDLES = {
        "wheelcollider", "wheelfrictioncurve", "wheelhit",
        "pxvehicle", "suspension"
    };

    @Override
    public void run() throws Exception {
        DataIterator data = currentProgram.getListing().getDefinedData(true);
        long matchingStrings = 0;
        long references = 0;
        while (data.hasNext() && !monitor.isCancelled()) {
            Data item = data.next();
            String representation = item.getDefaultValueRepresentation();
            if (representation == null || !matches(representation)) {
                continue;
            }
            matchingStrings++;
            println("STRING " + item.getAddress() + " " + representation);
            ReferenceIterator refs = currentProgram.getReferenceManager()
                    .getReferencesTo(item.getAddress());
            while (refs.hasNext()) {
                Reference ref = refs.next();
                Function function = getFunctionContaining(ref.getFromAddress());
                println("  REF " + ref.getFromAddress() + " -> "
                        + (function == null ? "<no function>" : function.getName()));
                references++;
            }
        }
        println("SUMMARY matching_defined_strings=" + matchingStrings
                + " references=" + references);
    }

    private boolean matches(String value) {
        String lower = value.toLowerCase();
        for (String needle : NEEDLES) {
            if (lower.contains(needle)) {
                return true;
            }
        }
        return false;
    }
}
