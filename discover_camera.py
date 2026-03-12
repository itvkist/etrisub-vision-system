from onvif import ONVIFCamera, exceptions
import netifaces
import socket

def get_local_ip():
    try:
        # Create a dummy socket connection to an external host
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error obtaining local IP: {e}")
        return None

def get_network_ip():
    ip = get_local_ip()
    if ip:
        # Assuming IP format is like 192.168.1.100
        base_ip = '.'.join(ip.split('.')[:-1])
        return base_ip
    else:
        print("Could not determine the current network IP.")
        return None

def discover_onvif_cameras(base_ip, start=1, end=255, port=80):
    cameras = []
    for i in range(start, end):
        ip = f"{base_ip}.{i}"
        try:
            # Attempt to connect to the camera
            cam = ONVIFCamera(ip, port, '', '')
            media_service = cam.create_media_service()
            profiles = media_service.GetProfiles()
            if profiles:
                cameras.append(ip)
                print(f"ONVIF camera found at {ip}")
        except exceptions.ONVIFError:
            continue
        except socket.timeout:
            continue

    return cameras

if __name__ == "__main__":
    base_ip = get_network_ip()
    if base_ip:
        print(f"Scanning network: {base_ip}.x for ONVIF cameras...")
        discovered_cameras = discover_onvif_cameras(base_ip)
        if discovered_cameras:
            print("Discovered ONVIF cameras:")
            for camera_ip in discovered_cameras:
                print(camera_ip)
        else:
            print("No ONVIF cameras found.")
    else:
        print("Unable to determine base IP for network scanning.")
