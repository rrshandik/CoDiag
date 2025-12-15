import xml.etree.ElementTree as ET

#load file .rdf
tree = ET.parse("tobonto_rev.rdf")
root = tree.getroot()

namespaces = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "swrl": "http://www.w3.org/2003/11/swrl#"
}

#namespace tobonto
ontology_ns = None
for elem in root.iter():
    if "terdiagnosis" in elem.tag:
        ontology_ns = elem.tag.split("}")[0].strip("{")
        break

ontology_ns = ontology_ns or "http://www.semanticweb.org/asus/ontologies/2025/11/tobonto"


#parsing individu
gejala_relasi = []

individual_elements = root.findall(".//owl:NamedIndividual", namespaces=namespaces)
if not individual_elements:
    individual_elements = root.findall(".//rdf:Description", namespaces=namespaces)

for individual in individual_elements:
    tipe_tags = individual.findall(".//rdf:type", namespaces=namespaces)
    for t in tipe_tags:
        tipe_val = t.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
        if tipe_val.endswith("gejala"):
            gejala_nama = individual.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "").split("#")[-1]
            terdiagnosis_tags = individual.findall(f".//{{{ontology_ns}}}terdiagnosis")
            for t in terdiagnosis_tags:
                target = t.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "").split("#")[-1]
                gejala_relasi.append((target, gejala_nama))

# buat dictionary mapping target -> gejala-gejala
individuals_dict = {}
for target, gejala in gejala_relasi:
    if target not in individuals_dict:
        individuals_dict[target] = []
    individuals_dict[target].append(gejala)

# ubah ke list of dict
individuals_list = [{"nama": nama, "gejala": gejala_list} for nama, gejala_list in individuals_dict.items()]


#parsing SWRL-nya
rule_list = []
r_value = 0

while r_value < 40:  #tobonto ada 29 rule
    r_value += 1
    r_label = f"R{r_value:03d}"

    individu_dict = {'rule': r_label}

    # cari rule berdasarkan rdfs:label
    target_rule = None
    for description in root.findall(".//rdf:Description", namespaces=namespaces):
        label_element = description.find("rdfs:label", namespaces=namespaces)
        if label_element is not None and label_element.text == r_label:
            target_rule = description
            break

    if target_rule is not None:
        def extract_value(uri):
            return uri.split('#')[-1] if uri else "Not found"

        # ambil gejala (dari argument2)
        nama_gejala = target_rule.find(
            ".//swrl:argument2[@rdf:resource]", namespaces=namespaces)
        individu_dict['gejala'] = extract_value(
            nama_gejala.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')) if nama_gejala is not None else "Not found"

        # ambil nama penyakit/hama (dari head)
        nama_serangan = target_rule.find(
            ".//swrl:head/rdf:Description/rdf:first/rdf:Description/swrl:argument1[@rdf:resource]",
            namespaces=namespaces)
        individu_dict['nama'] = extract_value(
            nama_serangan.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')) if nama_serangan is not None else "Not found"

        # ambil skor (float)
        risk_value = target_rule.find(
            ".//swrl:argument2[@rdf:datatype='http://www.w3.org/2001/XMLSchema#float']",
            namespaces=namespaces)
        individu_dict['skor'] = risk_value.text if risk_value is not None else "Not found"

        rule_list.append(individu_dict)
    else:
        continue


print("Individuals List:")
for i in individuals_list:
    print(i)

print("\nRule List:")
for r in rule_list:
    print(r)
