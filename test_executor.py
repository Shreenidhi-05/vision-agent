import executor

print("FILE:", executor.__file__)
print("DIR:", dir(executor))
print("HAS ACTIONEXECUTOR:", hasattr(executor, "ActionExecutor"))