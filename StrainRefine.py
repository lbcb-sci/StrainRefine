import argparse
import logging
import json
from pathlib import Path
import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import warnings
from sklearn.exceptions import DataConversionWarning
import os
import pickle

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() 
    ]
)

def read_reference_scores(filename):
    with open(filename, 'rb') as file:
        reference_scores = pickle.load(file)
    return reference_scores

def read_contig_info(filename):
    with open(filename, 'rb') as file:
        contig_info = pickle.load(file)
    return contig_info

def load_dict_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def species_split(U, NU, species_class, genomes, SS_info_json):
    SS_info = load_dict_from_json(SS_info_json)

    species = list(set(species_class.values()))

    U_species = {}
    NU_species = {}
    
    for s in species:
        U_species[s] = {}
        NU_species[s] = {}

    for read_id, value_list in U.items():
        s = species_class[read_id]
        U_species[s][read_id] = value_list

    for read_id, value_list in NU.items():
        s = species_class[read_id]
        new_value_list = [[], [], [], 0]
        for i, ind in enumerate(value_list[0]):
            s_v = SS_info[genomes[ind].split('|')[1]]
            if s_v == s:
                tmp = new_value_list[0]
                tmp.append(ind)
                new_value_list[0] = tmp

                tmp = new_value_list[1]
                tmp.append(value_list[1][i])
                new_value_list[1] = tmp

                tmp = new_value_list[2]
                tmp.append(value_list[2][i])
                new_value_list[2] = tmp

                new_value_list[3] = max(new_value_list[2])

        NU_species[s][read_id] = new_value_list

    return U_species, NU_species

    


def species_identification_with_thresholds(U, NU, genomes, species_count, SS_info_json,
                                           min_read_count=5, min_mean_score=0.6, low_count_cap=30):

    logging.info("======Species identification======")
    logging.info(
        "Starting species identification with thresholds: "
        f"min_read_count={min_read_count}, "
        f"min_mean_score={min_mean_score}, "
        f"low_count_cap={low_count_cap}"
    )

    SS_info = load_dict_from_json(SS_info_json)
    all_mappings = {**U, **NU}

    logging.info(
        f"Processing {len(all_mappings)} reads "
        f"({len(U)} uniquely mapped, {len(NU)} ambiguously mapped)"
    )
    
    def get_species_scores_for_read(read_id):
        species_scores = {}
        values = all_mappings[read_id]
        
        for i, genome_idx in enumerate(values[0]):
            genome = genomes[genome_idx]
            strain_taxid = genome.split('|')[1]
            species = SS_info[strain_taxid]
            score = values[2][i]
            
            if species not in species_scores:
                species_scores[species] = []
            species_scores[species].append((score, strain_taxid))
        
        return species_scores
    
    species_class = {}
    initial_ties = 0

    for read_id in all_mappings.keys():
        species_scores = get_species_scores_for_read(read_id)
        
        best_species = None
        best_score = -1
        candidates = []
        
        for species, scores in species_scores.items():
            max_score = max(s[0] for s in scores)
            if max_score > best_score:
                best_score = max_score
                best_species = species
                candidates = [species]
            elif max_score == best_score:
                candidates.append(species)
        
        if len(candidates) > 1:
            initial_ties += 1
            best_species = random.choice(candidates)
            logging.debug(
                f"Initial tie for read {read_id}: candidates={candidates}, chosen={best_species}"
            )
        
        species_class[read_id] = best_species

    logging.info(
        f"Initial species assignment complete. "
        f"Reads with tied best species score: {initial_ties}"
    )
    
    stable = False
    max_iterations = 100
    iteration = 0
    
    while not stable and iteration < max_iterations:
        iteration += 1
        species_class_old = species_class.copy()
        
        species_read_counts = {}
        species_scores = {}
        
        for read_id, species in species_class.items():
            if species not in species_read_counts:
                species_read_counts[species] = 0
                species_scores[species] = []
            
            species_read_counts[species] += 1
            
            species_score_dict = get_species_scores_for_read(read_id)
            if species in species_score_dict:
                best_score = max(s[0] for s in species_score_dict[species])
                species_scores[species].append(best_score)
        
        species_mean_scores = {
            s: np.mean(species_scores[s]) if species_scores[s] else 0
            for s in species_scores
        }
        
        threshold_read_count = min_read_count
        threshold_mean_score = min_mean_score
        unreliable_species = set()
        
        for s in species_read_counts:
            if (
                species_read_counts[s] < threshold_read_count or
                (
                    species_mean_scores[s] < threshold_mean_score and
                    species_read_counts[s] < low_count_cap
                )
            ):
                unreliable_species.add(s)

        logging.info(
            f"Iteration {iteration}: "
            f"{len(species_read_counts)} active species, "
            f"{len(unreliable_species)} unreliable species"
        )

        if unreliable_species:
            logging.debug(
                f"Iteration {iteration} unreliable species: "
                + ", ".join(
                    [
                        f"{s}(count={species_read_counts[s]}, mean={species_mean_scores[s]:.4f})"
                        for s in sorted(unreliable_species)
                    ]
                )
            )
        
        reassigned_from_unreliable = 0
        reassigned_by_tie_break = 0
        
        for read_id in all_mappings.keys():
            current_species = species_class[read_id]
            
            if current_species in unreliable_species:
                species_score_dict = get_species_scores_for_read(read_id)
                
                species_sorted = sorted(
                    species_score_dict.items(),
                    key=lambda x: max(s[0] for s in x[1]),
                    reverse=True
                )
                
                found_reliable = False
                for species, _ in species_sorted:
                    if species not in unreliable_species:
                        if species_class[read_id] != species:
                            logging.debug(
                                f"Iteration {iteration}: read {read_id} reassigned "
                                f"from unreliable species {species_class[read_id]} to {species}"
                            )
                            reassigned_from_unreliable += 1
                        species_class[read_id] = species
                        found_reliable = True
                        break
                
            else:
                species_score_dict = get_species_scores_for_read(read_id)
                current_score = max(s[0] for s in species_score_dict[current_species])
                
                max_score = -1
                tied_species = []
                for species, scores in species_score_dict.items():
                    sp_max = max(s[0] for s in scores)
                    if sp_max > max_score:
                        max_score = sp_max
                        tied_species = [species]
                    elif sp_max == max_score and species != current_species:
                        tied_species.append(species)
                
                if tied_species:
                    reliable_tied = [s for s in tied_species if s not in unreliable_species]
                    if reliable_tied:
                        best_tied = max(
                            reliable_tied,
                            key=lambda s: species_read_counts.get(s, 0)
                        )
                        if species_read_counts.get(best_tied, 0) > species_read_counts.get(current_species, 0):
                            logging.debug(
                                f"Iteration {iteration}: read {read_id} tie-resolved "
                                f"from {current_species} to {best_tied}"
                            )
                            species_class[read_id] = best_tied
                            reassigned_by_tie_break += 1
        
        stable = (species_class == species_class_old)

        changed_reads = sum(
            1 for read_id in species_class if species_class[read_id] != species_class_old[read_id]
        )

        logging.info(
            f"Iteration {iteration} summary: "
            f"{changed_reads} reads changed species assignment, "
            f"{reassigned_from_unreliable} reassigned from unreliable species, "
            f"{reassigned_by_tie_break} reassigned by tie-breaking, "
            f"stable={stable}"
        )

    final_species_counts = {}
    for species in species_class.values():
        final_species_counts[species] = final_species_counts.get(species, 0) + 1

    logging.info(
        f"Species identification finished after {iteration} iterations. "
        f"Final number of assigned species: {len(final_species_counts)}"
    )

    logging.debug(
        "Final species counts: "
        + ", ".join(
            [f"{s}:{c}" for s, c in sorted(final_species_counts.items(), key=lambda x: x[1], reverse=True)]
        )
    )

    if not stable:
        logging.warning(
            f"Species identification reached max_iterations={max_iterations} before convergence"
        )

    logging.info("==========================================")
    
    return species_class


def mapping_output(mapping_class_path, predictions_mapping):

    reduced = []
    if mapping_class_path is not None:
        f = open(mapping_class_path, "w")
        for contig, class_label in predictions_mapping.items():
            reduced.append(class_label)
            f.write(f"{contig} : {class_label}\n")
        f.close()

    logging.info("Mapping classification written in the file.")

def calculate_mapping_class(paf_path, species_strain_info, mapping_class_path=None, beta=2):
    logging.info("Mapping information extraction.")

    predictions_mapping = {}
    predictions_mapping_vcg = {}
    predictions_mapping_vcg_count = {}
    U = {}
    NU = {}

    genomes = {}
    genomes_names = {}
    genomes_id = 0
    genomes_list = []
    species_count = {}

    with open(paf_path, "r") as f:
        line = f.readline()
        while line != '':

            parts = line.split()
            read_id = parts[0]
        
            length_q = int(parts[3].strip()) - int(parts[2].strip())
            length_t = int(parts[8].strip()) - int(parts[7].strip())
            length_a = int(parts[10].strip())

            length = max(length_t, length_q)
            #nm = int(parts[9].strip().split(':')[-1])
            nm = int(parts[9].strip())
            value_cig = float(nm) / float(length)
            # print(value_cig)
            # value_cig = float(nm) - 4*(length_a-nm)

            ref_prediction = parts[5].strip()
            taxid = ref_prediction.split('|')[1]
            species_taxid = species_strain_info[taxid]

            if species_taxid in species_count:
                species_count[species_taxid] += 1
            else:
                species_count[species_taxid] = 1

            if ref_prediction not in genomes_list:
                genomes[genomes_id] = ref_prediction
                genomes_names[ref_prediction] = genomes_id
                genomes_id += 1
                genomes_list.append(ref_prediction)
            ref_id = genomes_names[ref_prediction]

            if read_id in predictions_mapping_vcg:

                if ref_prediction in predictions_mapping_vcg[read_id]:
                    if predictions_mapping_vcg[read_id][ref_prediction] < float(value_cig):
                        predictions_mapping_vcg[read_id][ref_prediction] = float(value_cig)
                    predictions_mapping_vcg_count[read_id][ref_prediction] += 1
                else:
                    predictions_mapping_vcg[read_id][ref_prediction] = float(value_cig)
                    predictions_mapping_vcg_count[read_id][ref_prediction] = 1
                    
            else:
                predictions_mapping_vcg[read_id] = {ref_prediction:float(value_cig)}
                predictions_mapping_vcg_count[read_id] = {ref_prediction:1}

            # print(predictions_mapping_vcg)
            line = f.readline()

    for read_id, candidates in predictions_mapping_vcg.items():

        max_values = max(list(candidates.values()))

        for candidate,value_cig in candidates.items():
            ref_id = genomes_names[candidate]
            if (read_id not in U) and (read_id not in NU):
                U[read_id] = [[ref_id], [value_cig], [value_cig], value_cig]
                predictions_mapping[read_id] = candidate
                continue

            if value_cig == max_values:
                predictions_mapping[read_id] = candidate

            if read_id in U:
                if ref_id in U[read_id][0]:
                    continue
                NU[read_id] = U[read_id]
                del U[read_id]  

            if ref_id in NU[read_id][0]:
                continue

            NU[read_id][0].append(ref_id)
            NU[read_id][1].append(value_cig)
            NU[read_id][2].append(value_cig)
            if value_cig > NU[read_id][3]:
                NU[read_id][3] = float(value_cig)

    if mapping_class_path is not None:
        mapping_output(mapping_class_path, predictions_mapping)

    return U, NU, genomes, species_count


def pathoscope_redistribution(NU, genomes):
    G = len(genomes)

    pi = [1./G for _ in genomes]

        
    for j in NU:
        z = NU[j] 
        ind = z[0] 
        pitmp = [pi[k] for k in ind]      
        xtmp = [1.*pitmp[k]*z[2][k] for k in range(len(ind))] 
            
        xsum = sum(xtmp)

        if xsum == 0:
            xnorm = [0.0 for k in xtmp]
        else:
            xnorm = [1.*k/xsum for k in xtmp]            

        NU[j][2] = xnorm  

    return NU

def find_medoid_and_avg_distance(cluster_indices, dist_matrix):
    if len(cluster_indices) == 1:
        return cluster_indices[0], 0.0 
    
    sub_matrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
    #print(sub_matrix)
    upper = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
    mean_distance = upper.mean()
    medoid_idx = cluster_indices[np.argmin(sub_matrix.mean(axis=1))]
    return medoid_idx, mean_distance
        
def initialize_clustering_output_dir(clustering_out):
    os.makedirs(clustering_out, exist_ok=True)

    return {
        "clusters_file": open(os.path.join(clustering_out, "clusters.txt"), "w"),
        "representatives_file": open(os.path.join(clustering_out, "representatives.txt"), "w"),
        "representatives_global": [],
        "avg_distances_global": [],
    }

def build_species_ref_dict(genomes, species_strain_info):
    species_ref_dict = {}
    references = list(set(genomes.values()))

    for ref in references:
        taxid = ref.split('|')[1]
        species_taxid = species_strain_info[taxid]
        species_ref_dict.setdefault(species_taxid, []).append(ref)

    return species_ref_dict

def collect_species_read_data(all_mappings, genomes):
    genome_read_dict = {}
    reads_index_dict = {}
    classified = []
    reference_scores = {}
    ambigous_refs_count = {}
    ambigous_refs_reads = {}
    ambigous_species_count = 0
    ref_count = {}

    for _, name in genomes.items():
        genome_read_dict[name] = [0] * len(all_mappings)
        ambigous_refs_count[name] = 0
        ambigous_refs_reads[name] = []
        ref_count[name] = 0

    for idx, read_id in enumerate(all_mappings.keys()):
        reads_index_dict[read_id] = idx

    for read_id, value_list in all_mappings.items():
        results = value_list[2]
        results_v1 = value_list[1]

        w = [i for i, x in enumerate(results) if x == max(results)]
        p = [value_list[0][i] for i in w]
        genome_list = [genomes[i] for i in p]

        if len(genome_list) == 1:
            genome = genome_list[0]
            classified.append((read_id, genome))

            reference_scores.setdefault(genome, [[], []])
            reference_scores[genome][0].append(read_id)
            reference_scores[genome][1].append(max(results))

            ref_count[genome] += 1
            ambigous_refs_count[genome] += 1
            genome_read_dict[genome][reads_index_dict[read_id]] = 1

        else:
            for ref in genome_list:
                ambigous_refs_count[ref] += 1
                ambigous_refs_reads[ref].append(read_id)
                genome_read_dict[ref][reads_index_dict[read_id]] = 1

            ambigous_species_count += 1

    return {
        "genome_read_dict": genome_read_dict,
        "reads_index_dict": reads_index_dict,
        "classified": classified,
        "reference_scores": reference_scores,
        "ambigous_refs_count": ambigous_refs_count,
        "ambigous_refs_reads": ambigous_refs_reads,
        "ambigous_species_count": ambigous_species_count,
        "ref_count": ref_count,
    }

def resolve_ambiguous_reads(all_mappings, genomes, ambigous_refs_count, classified, reference_scores, ref_count):
    new_class = {}

    for ref in genomes.values():
        new_class[ref] = 0

    for read_id, value_list in all_mappings.items():
        results = value_list[2]
        results_v1 = value_list[1]

        w = [i for i, x in enumerate(results) if x == max(results)]
        p = [value_list[0][i] for i in w]
        genome_list = [genomes[i] for i in p]

        if len(genome_list) <= 1:
            continue

        best_ref = genome_list[0]
        best_count = -1

        for ref in genome_list:
            c = ambigous_refs_count[ref]
            if c > best_count:
                best_ref = ref
                best_count = c

        classified.append((read_id, best_ref))
        new_class[best_ref] += 1
        ref_count[best_ref] += 1

        reference_scores.setdefault(best_ref, [[], []])
        reference_scores[best_ref][0].append(read_id)
        reference_scores[best_ref][1].append(max(results_v1))

    return classified, reference_scores, ref_count, new_class

def cluster_species_references(species_id, genome_read_dict, eps_value):
    new_genome_read_dict = {ref: arr for ref, arr in genome_read_dict.items() if 1 in arr}

    if len(new_genome_read_dict) == 0:
        return {
            "filtered_genome_read_dict": {},
            "clusters": [],
            "cluster_representatives": {},
        }

    ref_ids = list(new_genome_read_dict.keys())
    arrays = np.array(list(new_genome_read_dict.values()))
    dist_matrix = pairwise_distances(arrays, metric='jaccard')

    db = DBSCAN(metric='precomputed', eps=eps_value, min_samples=1)
    labels = db.fit_predict(dist_matrix)

    representatives = {}
    avg_distances = {}
    clusters = []
    cluster_representatives = {}

    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue

        cluster_indices = np.where(labels == label)[0]
        medoid_idx, avg_dist = find_medoid_and_avg_distance(cluster_indices, dist_matrix)
        medoid = ref_ids[medoid_idx]

        representatives[label] = medoid
        avg_distances[label] = avg_dist

    for cluster_id, medoid in representatives.items():
        cluster_refs = [ref_ids[i] for i, val in enumerate(labels) if val == cluster_id]
        ordered_cluster = [medoid] + [ref for ref in cluster_refs if ref != medoid]
        clusters.append(ordered_cluster)

        for ref in cluster_refs:
            cluster_representatives[ref] = medoid

    return {
        "filtered_genome_read_dict": new_genome_read_dict,
        "clusters": clusters,
        "cluster_representatives": cluster_representatives,
    }

def summarize_cluster_support(clusters, reference_scores, high_score_threshold):
    ref_high_scores = {}
    ref_high_scores_global = {}
    low_score_reads_global = []
    assigned_reads = 0

    logging.info("======Cluster support summary======")
    for i, cluster in enumerate(clusters, start=1):
        logging.info(f"Cluster {i}:")
        for ref in cluster:
            if ref in reference_scores:
                scores = reference_scores[ref]
                high_scores = [s for s in scores[1] if s >= high_score_threshold]
                low_score_reads = [scores[0][k] for k in range(len(scores[0])) if scores[1][k] < high_score_threshold]

                low_score_reads_global.extend(low_score_reads)
                ref_high_scores[ref] = len(high_scores)
                ref_high_scores_global[ref] = len(high_scores) / len(scores[1]) if len(scores[1]) > 0 else 0.0

                d = len(high_scores) if len(high_scores) > 0 else 1
                assigned_reads += len(scores[0])

                logging.info(
                    "Reference: {}, Assigned Reads: {}, Average Score: {:.4f}, High Scores: {}".format(
                        ref, len(scores[0]), sum(high_scores) / d, len(high_scores)
                    )
                )
            else:
                ref_high_scores[ref] = 0
                ref_high_scores_global[ref] = 0.0
                logging.info("Reference: {}, Assigned Reads: 0, Average Score: 0.0000, High Scores: 0".format(ref))

    low_score_reads_global = list(set(low_score_reads_global))

    logging.info("Total low score reads globally: {}".format(len(low_score_reads_global)))
    logging.info("==========================================")

    # print(ref_high_scores)

    return {
        "ref_high_scores": ref_high_scores,
        "ref_high_scores_global": ref_high_scores_global,
        "low_score_reads_global": low_score_reads_global,
        "assigned_reads": assigned_reads,
    }


def choose_reference_reassignments(clusters, clusters_species, s_genome_read_dicts,
                                   ref_high_scores, ref_high_scores_global, cfg):
    changes = {}

    logging.info("======Reference reassignment summary======")

    for i, cluster in enumerate(clusters):
        species = clusters_species[i]

        for ref in cluster:
            if ref not in ref_high_scores:
                ref_high_scores[ref] = 0
            if ref not in ref_high_scores_global:
                ref_high_scores_global[ref] = 0.0

            #print("Reference: {}, High Scores: {}".format(ref, ref_high_scores[ref]))

            weak_ref = (
                (ref_high_scores[ref] < cfg["min_high_score_reads"]) or
                (
                    ref_high_scores[ref] >= cfg["min_high_score_reads"] and
                    ref_high_scores[ref] < cfg["mid_high_score_reads"] and
                    ref_high_scores_global[ref] < cfg["min_high_score_fraction"]
                )
            )

            if not weak_ref:
                continue

            genome_read_dict = s_genome_read_dicts[species]
            keys = list(genome_read_dict.keys())
            arrays = np.array(list(genome_read_dict.values()))
            dist_matrix = pairwise_distances(arrays, metric='jaccard')

            ref_i = keys.index(ref)
            row = dist_matrix[ref_i].copy()
            row[ref_i] = np.inf
            sorted_idx = np.argsort(row)

            for j in sorted_idx:
                cand = keys[j]

                if cand == ref:
                    continue
                if cand in changes:
                    continue
                if cand not in ref_high_scores or cand not in ref_high_scores_global:
                    continue

                strong_cand = (
                    (ref_high_scores[cand] > cfg["min_high_score_reads"]) or
                    (ref_high_scores_global[cand] > cfg["min_high_score_fraction"])
                )

                if strong_cand:
                    changes[ref] = cand
                    closest_distance = row[j]
                    logging.info(
                        "Cluster {} - Reference {} changed to {} based on clustering with distance {:.4f}".format(
                            i + 1, ref, cand, closest_distance
                        )
                    )
                    break

    # print(changes)
    logging.info("==========================================")
    return changes

def get_ref_count(classified, changes):
    ref_count = {}
    for read_id, ref in classified:
        final_ref = changes[ref] if ref in changes else ref
        ref_count[final_ref] = ref_count.get(final_ref, 0) + 1
    return ref_count

def recompute_cluster_representatives_by_count(clusters, ref_count):
    changes = {}
    representatives_new = []

    updated_clusters = []

    for cluster in clusters:
        max_count = 0
        max_ref = ""

        for ref in cluster:
            count = ref_count[ref] if ref in ref_count else 0
            if count > max_count:
                max_count = count
                max_ref = ref

        if max_ref != "":
            for ref in cluster:
                changes[ref] = max_ref

        representatives_new.append(max_ref)
        updated_clusters.append([max_ref] + [ref for ref in cluster if ref != max_ref])

    return changes, representatives_new, updated_clusters

def write_final_cluster_outputs(clustering_out, updated_clusters, representatives_new):
    with open(os.path.join(clustering_out, "clusters.txt"), "w") as f_clusters:
        for cluster in updated_clusters:
            f_clusters.write(" ".join(cluster).strip() + "\n")

    with open(os.path.join(clustering_out, "representatives.txt"), "w") as f_rep:
        for rep in representatives_new:
            f_rep.write(rep + "\n")

def write_final_assignments(classified, initial_changes, cluster_changes, output_path):
    with open(output_path, "w") as f:
        for read_id, ref in classified:
            if ref in cluster_changes:
                final_ref = cluster_changes[ref]
            elif ref in initial_changes:
                final_ref = initial_changes[ref]
            else:
                final_ref = ref

            if final_ref == "":
                final_ref = ref

            f.write(f"{read_id} : {final_ref}\n")


def run(args):
    logging.info("Parameters:")
    logging.info(f"Strain-Species info JSON file path: {args.strain_species_info}")
    logging.info(f"Input PAF file path: {args.paf_path}")
    logging.info(f"Final classification labels file path: {args.read_class_output}")

    cfg = {
        "cluster_eps": args.cluster_eps,
        "high_score_threshold": args.high_score_threshold,
        "min_high_score_reads": args.min_high_score_reads,
        "mid_high_score_reads": args.mid_high_score_reads,
        "min_high_score_fraction": args.min_high_score_fraction,
        "species_min_read_count": args.species_min_read_count,
        "species_min_mean_score": args.species_min_mean_score,
        "species_low_count_cap": args.species_low_count_cap,
    }

    initialize_clustering_output_dir(args.clustering_out)

    species_strain_info = load_dict_from_json(args.strain_species_info)
    U, NU, genomes, species_count = calculate_mapping_class(
        args.paf_path, species_strain_info, mapping_class_path=None, beta=0.5
    )

    species_class = species_identification_with_thresholds(
        U, NU, genomes, species_count, args.strain_species_info,
        min_read_count=cfg["species_min_read_count"],
        min_mean_score=cfg["species_min_mean_score"],
        low_count_cap=cfg["species_low_count_cap"]
    )
    U_species, NU_species = species_split(U, NU, species_class, genomes, args.strain_species_info)

    species_ref_dict = build_species_ref_dict(genomes, species_strain_info)
    species = list(set(species_class.values()))

    s_genome_read_dicts = {}
    clusters = []
    clusters_species = []
    cluster_representatives = {}
    classified = []
    reference_scores = {}
    assigned_reads = 0

    for s in species:
        logging.info("Species {} has {} references, eps value for clustering: {}".format(
            s, len(species_ref_dict[s]), cfg["cluster_eps"])
        )

        NU_result = pathoscope_redistribution(NU_species[s], genomes)
        all_mappings = {**U_species[s], **NU_result}

        species_data = collect_species_read_data(all_mappings, genomes)

        classified.extend(species_data["classified"])
        reference_scores.update(species_data["reference_scores"])

        classified, reference_scores, _, _ = resolve_ambiguous_reads(
            all_mappings,
            genomes,
            species_data["ambigous_refs_count"],
            classified,
            reference_scores,
            species_data["ref_count"]
        )


        clustering_result = cluster_species_references(
            s,
            species_data["genome_read_dict"],
            cfg["cluster_eps"]
        )

        s_genome_read_dicts[s] = clustering_result["filtered_genome_read_dict"]
        clusters.extend(clustering_result["clusters"])
        clusters_species.extend([s] * len(clustering_result["clusters"]))
        cluster_representatives.update(clustering_result["cluster_representatives"])

    support_summary = summarize_cluster_support(
        clusters, reference_scores, cfg["high_score_threshold"]
    )
    assigned_reads = support_summary["assigned_reads"]

    changes = choose_reference_reassignments(
        clusters,
        clusters_species,
        s_genome_read_dicts,
        support_summary["ref_high_scores"],
        support_summary["ref_high_scores_global"],
        cfg
    )

    ref_count = get_ref_count(classified, changes)

    changes2, representatives_new, updated_clusters = recompute_cluster_representatives_by_count(
        clusters, ref_count
    )


    write_final_cluster_outputs(args.clustering_out, updated_clusters, representatives_new)
    write_final_assignments(
        classified,
        changes,
        changes2,
        args.read_class_output
    )

    logging.info("Total assigned reads: {}".format(assigned_reads))
   
  

def main():

    parser = argparse.ArgumentParser(description="MADRe.")

    parser.add_argument(
        "--paf_path", type=str, required=True,
        help="Path to the PAF file of assembly mapped to database."
    )

    parser.add_argument(
        "--strain_species_info", type=str, required=True,
        help="An additional parameter required if a custom database path is provided. JSON file with info about species taxid for every strain taxid in the database. If you want to use default one provide path to MADRe/database/taxids_species.json."
    )
    parser.add_argument(
        "--cluster_eps", type=float, default=0.8,
        help="DBSCAN eps value for clustering references within species (default=0.8)."
    )

    parser.add_argument(
        "--high_score_threshold", type=float, default=0.6,
        help="Score threshold used to count high-confidence assigned reads per reference (default=0.6)."
    )

    parser.add_argument(
        "--min_high_score_reads", type=int, default=5,
        help="Minimum number of high-score reads for a reference to be considered supported (default=5)."
    )

    parser.add_argument(
        "--mid_high_score_reads", type=int, default=10,
        help="Intermediate support threshold for references (default=10)."
    )

    parser.add_argument(
        "--min_high_score_fraction", type=float, default=0.8,
        help="Minimum fraction of high-score reads for moderately supported references (default=0.8)."
    )

    parser.add_argument(
        "--species_min_read_count", type=int, default=5,
        help="Minimum read count threshold for species reliability. Currently informative only unless species identification is refactored (default=5)."
    )

    parser.add_argument(
        "--species_min_mean_score", type=float, default=0.6,
        help="Minimum mean score threshold for species reliability. Currently informative only unless species identification is refactored (default=0.6)."
    )

    parser.add_argument(
        "--species_low_count_cap", type=int, default=30,
        help="Species with mean score below threshold and read count below this cap are treated as unreliable. Currently informative only unless species identification is refactored (default=30)."
    )


    parser.add_argument(
        "--read_class_output", type=str, default="read_classification.out",
        help="Path to the output file with classification labels for reads. (default=read_classification.out)" 
    )

    parser.add_argument(
    "--clustering_out", type=str, default="clustering_output",
    help="Path to the directory for clustering-related output files (default=clustering_output)."
)
    
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
