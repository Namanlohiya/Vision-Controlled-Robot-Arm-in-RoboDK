from robodk.robolink import Robolink

# Initialize RoboDK API
RDK = Robolink()

# Verify connection to RoboDK
if RDK.Connect():  # Changed from Connected() to Connect()
    print("Successfully connected to RoboDK!")
else:
    raise ConnectionError("Failed to connect to RoboDK. Please ensure RoboDK software is running.")
