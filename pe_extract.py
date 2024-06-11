import os
import pefile
import csv

# Prajnadeep
# 17-07-2021

# Create Header
IMAGE_DOS_HEADER = ["Name", "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr", "e_minalloc", "e_maxalloc", "e_ss",
                    "e_sp", "e_csum", "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid", "e_oeminfo",
                    "e_lfanew"]

FILE_HEADER = ["Machine", "NumberOfSections", "CreationYear", "PointerToSymbolTable",
               "NumberOfSymbols", "SizeOfOptionalHeader", "Characteristics"]

OPTIONAL_HEADER = ["Magic", "MajorLinkerVersion", "MinorLinkerVersion", "SizeOfCode", "SizeOfInitializedData",
                   "SizeOfUninitializedData", "AddressOfEntryPoint",
                   "BaseOfCode", "ImageBase", "SectionAlignment", "FileAlignment",
                   "MajorOperatingSystemVersion", "MinorOperatingSystemVersion",
                   "MajorImageVersion",
                   "MinorImageVersion",
                   "MajorSubsystemVersion",
                   "MinorSubsystemVersion",
                   "SizeOfHeaders",
                   "CheckSum",
                   "SizeOfImage",
                   "Subsystem",
                   "DllCharacteristics",
                   "SizeOfStackReserve",
                   "SizeOfStackCommit",
                   "SizeOfHeapReserve",
                   "SizeOfHeapCommit",
                   "LoaderFlags",
                   "NumberOfRvaAndSizes"]


def file_creation_year(seconds):
    return 1970 + ((int(seconds) / 86400) / 365)


def extract_dos_header(pe, file_name):
    IMAGE_DOS_HEADER_data = [0 for i in range(18)]

    try:
        IMAGE_DOS_HEADER_data = [
            file_name,
            pe.DOS_HEADER.e_magic,
            pe.DOS_HEADER.e_cblp,
            pe.DOS_HEADER.e_cp,
            pe.DOS_HEADER.e_crlc,
            pe.DOS_HEADER.e_cparhdr,
            pe.DOS_HEADER.e_minalloc,
            pe.DOS_HEADER.e_maxalloc,
            pe.DOS_HEADER.e_ss,
            pe.DOS_HEADER.e_sp,
            pe.DOS_HEADER.e_csum,
            pe.DOS_HEADER.e_ip,
            pe.DOS_HEADER.e_cs,
            pe.DOS_HEADER.e_lfarlc,
            pe.DOS_HEADER.e_ovno,
            pe.DOS_HEADER.e_oemid,
            pe.DOS_HEADER.e_oeminfo,
            pe.DOS_HEADER.e_lfanew]

    except pefile.PEFormatError:
        print("Exception")
        pass

    return IMAGE_DOS_HEADER_data


def extract_features(pe, file_name):
    try:
        IMAGE_DOS_HEADER_data = extract_dos_header(pe, file_name)

    except Exception:
        pass

    else:
        FILE_HEADER_data = [pe.FILE_HEADER.Machine,
                            pe.FILE_HEADER.NumberOfSections,
                            file_creation_year(pe.FILE_HEADER.TimeDateStamp),
                            pe.FILE_HEADER.PointerToSymbolTable,
                            pe.FILE_HEADER.NumberOfSymbols,
                            pe.FILE_HEADER.SizeOfOptionalHeader,
                            pe.FILE_HEADER.Characteristics]

        OPTIONAL_HEADER_data = [pe.OPTIONAL_HEADER.Magic,
                                pe.OPTIONAL_HEADER.MajorLinkerVersion,
                                pe.OPTIONAL_HEADER.MinorLinkerVersion,
                                pe.OPTIONAL_HEADER.SizeOfCode,
                                pe.OPTIONAL_HEADER.SizeOfInitializedData,
                                pe.OPTIONAL_HEADER.SizeOfUninitializedData,
                                pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                                pe.OPTIONAL_HEADER.BaseOfCode,
                                pe.OPTIONAL_HEADER.ImageBase,
                                pe.OPTIONAL_HEADER.SectionAlignment,
                                pe.OPTIONAL_HEADER.FileAlignment,
                                pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
                                pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
                                pe.OPTIONAL_HEADER.MajorImageVersion,
                                pe.OPTIONAL_HEADER.MinorImageVersion,
                                pe.OPTIONAL_HEADER.MajorSubsystemVersion,
                                pe.OPTIONAL_HEADER.MinorSubsystemVersion,
                                pe.OPTIONAL_HEADER.SizeOfHeaders,
                                pe.OPTIONAL_HEADER.CheckSum,
                                pe.OPTIONAL_HEADER.SizeOfImage,
                                pe.OPTIONAL_HEADER.Subsystem,
                                pe.OPTIONAL_HEADER.DllCharacteristics,
                                pe.OPTIONAL_HEADER.SizeOfStackReserve,
                                pe.OPTIONAL_HEADER.SizeOfStackCommit,
                                pe.OPTIONAL_HEADER.SizeOfHeapReserve,
                                pe.OPTIONAL_HEADER.SizeOfHeapCommit,
                                pe.OPTIONAL_HEADER.LoaderFlags,
                                pe.OPTIONAL_HEADER.NumberOfRvaAndSizes]

        return IMAGE_DOS_HEADER_data + FILE_HEADER_data + OPTIONAL_HEADER_data


def extractDatatoCSV():
    output = "instance/testCSV/test.csv"
    success = False

    # Directory name
    directory = 'instance/downloads'

    # Create CSV
    f = open(output, 'wt')
    writer = csv.writer(f)
    writer.writerow(IMAGE_DOS_HEADER + FILE_HEADER + OPTIONAL_HEADER)

    print("Extracting features")
    # iterate over files in directory
    for file in os.listdir(directory):
        try:
            pe = pefile.PE('instance/downloads/' + file)
            print(file)
            if pe.DOS_HEADER.e_magic == int(0x5a4d) and pe.NT_HEADERS.Signature == int(0x4550):
                features = extract_features(pe, file)
                writer.writerow(features)
                pe.close()
                success = True

        except Exception as e:
            print("Exception while opening and writing CSV file: ", e)
            pass

    f.close()
    print("Features saved to ", output)

    return success
