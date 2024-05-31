#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include <optional>
#include <chrono>

using namespace mlir;
using namespace affine;
using namespace llvm;

namespace mlir {
namespace affine {

#define GEN_PASS_DEF_AFFINETILINGWITHAI

#include "mlir/Dialect/Affine/Passes.h.inc"

} // namespace affine
} // namespace mlir

namespace {
class AffineTilingWithAI
  : public affine::impl::AffineTilingWithAIBase<AffineTilingWithAI> {
private:
  void runOnOperation() override {

    // Declare the variables
    long long int arithops = 0;
    long long int memaccess = 0;

    // auto start = std::chrono::steady_clock::now();

    // // End measuring time
    // auto end = std::chrono::steady_clock::now();

    // // Calculate and print the runtime
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // errs() << "Runtime: " << duration << " milliseconds\n";

    // Arithmetic intensity = A / B

    long long int arops = 0;
    getOperation()->walk([&](Operation *op)
    {
      if (op->getDialect()->getNamespace() == "arith")
        arops++;
    });
    errs() << "\nCount of Arithmetic Ops: " << arops << "\n";

    Operation *op = getOperation(); // Get the current operation being operated on

    // Handle Store Instructions
    long long int load = 0;
    op->walk([&](affine::AffineLoadOp  bodyOp)
    {
      // Check if the operation is a load
      errs() << "\nLoad Op: ";
      bodyOp->dump();
      // errs() << "LoadInst" << bodyOp;
      // errs() << "\n";
      load++;
    });
    errs() << "\nCount of load op: " << load << "\n";

    // Handle Load Instructions
    // op->walk([&](memref::LoadOp  bodyOp)
    // {
    // // Check if the operation is a load or store
    //   errs() << "Load Op: ";
    //   bodyOp->dump();
    //   errs() << "\n";
    //   // errs() << "LoadInst" << bodyOp;
    //   load++;
    // }); 
    // errs() << "Count of load op: " << load << "\n";

    // // Handle Store Instructions
    // int store = 0;
    // op->walk([&](memref::StoreOp  bodyOp)
    // {
    //   // Check if the operation is a load or store
    //   errs() << "\nStore Op: ";
    //   bodyOp->dump();
    //   // errs() << "StoreInst" << bodyOp;
    //   // errs() << "\n";
    //   store++;
    // });
    // errs() << "Count of store op: " << store << "\n";
    
    long long int store = 0;
    op->walk([&](affine::AffineStoreOp  bodyOp)
    {
    // Check if the operation is a load or store
      errs() << "\nStore Op: ";
      bodyOp->dump();
      errs() << "\n";
      // errs() << "StoreInst" << bodyOp;
      load++;
    }); 
    errs() << "Count of store op: " << store << "\n";

    // // Arithmetic Operations
    // int arops = 0;
    // op->walk([&](arith::AddIOp ArOp) // Handle integer addition
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::AddFOp ArOp) // Handle float addition
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::SubIOp ArOp) // Handle int subtraction
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::SubFOp ArOp) // Handle float subtraction
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::MulIOp ArOp) // Handle int multiplication
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::MulFOp ArOp) // Handle float multiplication
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::DivUIOp ArOp) // Handle unsigned int division
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::DivSIOp ArOp) // Handle signed int division
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::DivFOp ArOp) // Handle float division
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::RemSIOp ArOp) // Handle signed remainder operarion
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::RemUIOp ArOp) // Handle unsigned remainder operarion
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::RemSIOp ArOp) // Handle signed remainder operarion
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    // op->walk([&](arith::RemFOp ArOp) // Handle floating point remainder operarion
    // {
    //   errs() << "\nArithmetic Op: " << ArOp->getName() << "\n";
    //   ArOp->dump();
    //   errs() << "\n";
    //   arops++;
    // });

    int itr = 1;

    // Handle affine.for ops
    op->walk([&](affine::AffineForOp  bodyOp)
    {
      errs() << "\nLoop: " << bodyOp->getName() << "\n"; // affine.for
  
      auto i = bodyOp.getConstantLowerBound();
      auto j = bodyOp.getConstantUpperBound();
      // auto depth = bodyOp.getStep();
      // auto s = bodyOp.getBody();
      errs() << "Lower Bound: " << i << "\n";
      errs() << "Upper Bound: " << j << "\n";
      // errs() << "Step: " << depth << "\n";
      // errs() << "Body: " << s << "\n";
      itr *= (j-i);
    });

    errs() << "\nNo. of iterations: " << itr << "\n";

    arithops = arops * itr;
    memaccess = (load+store)*itr;

    errs() << "\nCount of Arithmetic Ops: " << arops << "\n";
    errs() << "\nCount of Memory References: " << (load+store) << "\n";
    errs() << "\nTotal number of Arithmeric Ops: " << arithops << "\n";
    errs() << "\nTotal number of Memory references: " << (memaccess*4) << "\n";

    float AI = arithops/memaccess;

    errs() << "\nArithmetic Intensity: " << AI << "\n";

    // Perform Loop Tiling

    getOperation()->walk([&](mlir::affine::AffineForOp op)
    {
      int tileSizes = 32; // Example tile sizes

      SmallVector<mlir::affine::AffineForOp, 7> loopNest;
      loopNest.push_back(op);
      // if (op.hasConstantLowerBound())
      // {
      //   int64_t lowerBound = op.getConstantLowerBound();
      //   int64_t upperBound = op.getConstantUpperBound();
      //   errs() << "Lower Bound: " << lowerBound << "\n";
      //   errs() << "Upper Bound: " << upperBound << "\n";
      // }

      // Perform pre-tiling checks
      if (failed(affine::tilePerfectlyNested(loopNest, tileSizes)))
      {
        return; // Handle pre-tiling check failures (e.g., emit warnings or // errors)
      }
    });
  } // runOnOperation
};
}