from sir import RunSimpleSimulationWithDots

testSim = RunSimpleSimulationWithDots();
testSim.add_simulation();
latentData = testSim.construct();

print("sir_simple_test completed. Latent data is: \n");
print(latentData[0]); print("\n ... \n"); print(latentData[len(latentData.keys())-1]);