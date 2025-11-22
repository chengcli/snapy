import argparse, os, shutil
from glob import glob
from subprocess import check_call
from numpy import sort

def CombineTimeseries(case, field, stamps, path="./", remove=False):
    print("Concatenating output field %s ..." % field)
    fname = ""
    for stamp in stamps:
        fname += path + "/%s.%s.%s.nc " % (case, field, stamp)
    target = "%s.%s.nc" % (case, field)

    if len(stamps) > 1:
        # check_call('ncrcat -h %s -o %s' % (fname, target), shell = True)
        check_call(
            "ncrcat -h %s/%s.%s.?????.nc -o %s/%s" % (path, case, field, path, target),
            shell=True,
        )
        if remove:
            for f in fname.split():
                os.remove(f)
    else:
        shutil.move(fname[:-1], target)


def CombineFields(case, fields, name, path="./"):
    fid = [int(f) for f in fields.split(",")]

    print("Combining output fields: ", fields, "to", name)
    ncout = "%s/%s-%s.nc" % (path, case, name)

    if len(fid) > 1:  # combining
        fname1 = path + "/%s.out%d.nc" % (case, fid[0])
        for i in fid[1:]:
            fname2 = path + "/%s.out%d.nc" % (case, i)
            check_call("ncks -A %s %s" % (fname2, fname1), shell=True)
            os.remove(fname2)
    else:  # renaming
        fname1 = path + "/%s.out%d.nc" % (case, fid[0])
    shutil.move(fname1, ncout)


def ParseOutputFields(path):
    files = glob(path + "/*.out*.[0-9][0-9][0-9][0-9][0-9].nc")
    cases = []
    fields = []
    stamps = []

    for fname in files:
        field, stamp, ext = os.path.basename(fname).split(".")[-3:]
        case = ".".join(os.path.basename(fname).split(".")[:-3])
        if case not in cases:
            cases.append(case)
        if field not in fields:
            fields.append(field)
        if stamp not in stamps:
            stamps.append(stamp)
    stamps = sorted(stamps)
    return cases, fields, stamps


def CombineFITS(case, output, path="./", remove=False):
    print("Combining FITS output ...")
    files = glob(path + "/%s.out?.[0-9][0-9][0-9][0-9][0-9].fits" % case)
    if len(files) == 0:
        return None
    if output != "None":
        fitsout = "%s-%s.fits" % (case, output)
    else:
        fitsout = "%s.fits" % case
    root = os.path.dirname(os.path.realpath(__file__))
    check_call(
        "%s/bin/fitsmerge -o %s -i %s/%s.out?.?????.fits" % (root, fitsout, path, case),
        shell=True,
    )
    if remove:
        for f in files:
            os.remove(f)
    return fitsout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_ids',
        help='comma-separated list of output ids to combine (e.g., "1,2,3")'
    )

    parser.add_argument(
        "-d", "--dir", default=".", help="directory of the simulation to combine"
    )
    parser.add_argument(
        "-o", "--output", default="main", help="combined output name"
    )
    parser.add_argument(
        "--no-remove", action="store_true", help="do not remove original files"
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="do not merge different fields"
    )
    args = vars(parser.parse_args())

    cases, fields, stamps = ParseOutputFields(args["dir"])
    fields.sort()

    for f in args["output_ids"].split(","):
        fname = 'out%s' % f
        if fname not in fields:
            print("Error: output id %s not found!" % f)
            exit(1)

    print("##########################")
    print("## Combine output files ##")
    print("##########################")
    for case in cases:
        print("Working on case %s..." % case)
        fitsout = CombineFITS(
            case, args["output"], path=args["dir"], remove=not args["no_remove"]
        )
        for field in fields:
            CombineTimeseries(
                case, field, stamps, remove=not args["no_remove"], path=args["dir"]
            )
        if not args["no_merge"]:
            CombineFields(case, args["output_ids"], args["output"], path=args["dir"])
    print("Done.\n")

if __name__ == "__main__":
    main()
