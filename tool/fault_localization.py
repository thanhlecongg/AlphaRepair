import os


def get_loc_file(bug_id, perfect):
    dirname = os.path.dirname(__file__)
    if perfect:
        loc_file = '../location/groundtruth/%s/%s' % (bug_id.split("-")[0].lower(), bug_id.split("-")[1])
    else:
        loc_file = '../location/ochiai/%s/%s.txt' % (bug_id.split("-")[0].lower(), bug_id.split("-")[1])
    loc_file = os.path.join(dirname, loc_file)
    if os.path.isfile(loc_file):
        return loc_file
    else:
        print(loc_file)
        return ""


# grab location info from bug_id given
# perfect fault localization returns 1 line, top n gets top n lines for non-perfect FL (40 = decoder top n)
def get_location(bug_id, perfect=True, top_n=40):
    source_dir = os.popen("defects4j export -p dir.src.classes -w /tmp/" + bug_id).readlines()[-1].strip() + "/"
    location = []
    location_dict = {}
    loc_file = get_loc_file(bug_id, perfect)
    if loc_file == "":
        return location
    if perfect:
        lines = open(loc_file, 'r').readlines()
        for loc_line in lines:
            loc_line = loc_line.split("||")[0]  # take first line in lump
            classname, line_id = loc_line.split(':')
            classname = ".".join(classname.split(".")[:-1])  # remove function name
            if '$' in classname:
                classname = classname[:classname.index('$')]
            file = source_dir + "/".join(classname.split(".")) + ".java"
            location.append((file, int(line_id) - 1))
    else:
        lines = open(loc_file, 'r').readlines()
        for loc_line in lines:
            loc_line = loc_line.split(",")[0]
            classname, line_id = loc_line.split("#")
            if '$' in classname:
                classname = classname[:classname.index('$')]
            file = source_dir + "/".join(classname.split(".")) + ".java"
            if file + line_id not in location_dict:
                location.append((file, int(line_id) - 1))
                location_dict[file + line_id] = 0
            else:
                print("Same Fault Location: {}, {}".format(file, line_id))
        pass

    return location[:top_n]
