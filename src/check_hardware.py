import openvino as ov


def main():
    print("🔍 Scanning system for OpenVINO-compatible hardware...\n")

    try:
        core = ov.Core()
        devices = core.available_devices

        if not devices:
            print("❌ No compatible devices found! Check your OpenVINO installation.")
            return

        for device in devices:
            try:
                # Ask OpenVINO for the plain-English name of the hardware
                device_name = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"✅ {device.ljust(6)} : {device_name}")
            except Exception as e:
                print(
                    f"⚠️ {device.ljust(6)} : Detected, but couldn't read full name ({e})"
                )

        print(
            "\nHardware check complete. If you see 'NPU' above, you are ready for Redactyl!"
        )

    except Exception as e:
        print(f"❌ Failed to initialize OpenVINO Core: {e}")


if __name__ == "__main__":
    main()
