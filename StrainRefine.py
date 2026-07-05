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

def compute_horizontal_coverage(ref_intervals, ref_name):
    return ref_intervals.get(ref_name, 0.0)

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

def species_split(U, NU, species_class, genomes, SS_info):
    species = set(species_class.values())
    genome_species = {ind: SS_info[name.split('|')[1]] for ind, name in genomes.items()}

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
            if genome_species[ind] == s:
                new_value_list[0].append(ind)
                new_value_list[1].append(value_list[1][i])
                new_value_list[2].append(value_list[2][i])
                new_value_list[3] = max(new_value_list[3], value_list[2][i])

        if not new_value_list[0]:
            logging.warning(f"Read {read_id} assigned to species {s} but has no mappings to it — skipping")
            continue

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
    
    all_species_scores = {
        read_id: get_species_scores_for_read(read_id)
        for read_id in all_mappings
    }
    all_max_species_scores = {
        read_id: {sp: max(s[0] for s in scores) for sp, scores in ssd.items()}
        for read_id, ssd in all_species_scores.items()
    }
    all_sorted_species = {
        read_id: sorted(max_scores.items(), key=lambda x: x[1], reverse=True)
        for read_id, max_scores in all_max_species_scores.items()
    }
    del all_species_scores

    species_class = {}
    initial_ties = 0

    for read_id in all_mappings.keys():
        max_scores = all_max_species_scores[read_id]

        best_species = None
        best_score = -1
        candidates = []

        for species, max_score in max_scores.items():
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

        species_read_counts = {}
        species_score_sums = {}
        species_score_counts = {}

        for read_id, species in species_class.items():
            species_read_counts[species] = species_read_counts.get(species, 0) + 1
            max_scores = all_max_species_scores[read_id]
            if species in max_scores:
                species_score_sums[species] = species_score_sums.get(species, 0.0) + max_scores[species]
                species_score_counts[species] = species_score_counts.get(species, 0) + 1

        species_mean_scores = {
            s: species_score_sums[s] / species_score_counts[s] if species_score_counts.get(s, 0) > 0 else 0.0
            for s in species_read_counts
        }
        
        unreliable_species = set()

        for s in species_read_counts:
            if (
                species_read_counts[s] < min_read_count or
                (
                    species_mean_scores[s] < min_mean_score and
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
                for species, _ in all_sorted_species[read_id]:
                    if species not in unreliable_species:
                        if species_class[read_id] != species:
                            logging.debug(
                                f"Iteration {iteration}: read {read_id} reassigned "
                                f"from unreliable species {species_class[read_id]} to {species}"
                            )
                            reassigned_from_unreliable += 1
                        species_class[read_id] = species
                        break

            else:
                max_scores = all_max_species_scores[read_id]
                current_score = max_scores[current_species]

                best_alt_score = -1
                best_alt_species = None
                for species, sp_max in max_scores.items():
                    if species == current_species or species not in species_read_counts or species in unreliable_species:
                        continue
                    if sp_max > best_alt_score or (
                        sp_max == best_alt_score and
                        species_read_counts.get(species, 0) > species_read_counts.get(best_alt_species, 0)
                    ):
                        best_alt_score = sp_max
                        best_alt_species = species

                if best_alt_species is not None:
                    if best_alt_score > current_score or (
                        best_alt_score == current_score and
                        species_read_counts.get(best_alt_species, 0) > species_read_counts.get(current_species, 0)
                    ):
                        logging.debug(
                            f"Iteration {iteration}: read {read_id} tie-resolved "
                            f"from {current_species} to {best_alt_species}"
                        )
                        species_class[read_id] = best_alt_species
                        reassigned_by_tie_break += 1
        
        changed_reads = reassigned_from_unreliable + reassigned_by_tie_break
        stable = (changed_reads == 0)

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

    U = {}
    NU = {}
    read_genome_idx = {}  # read_id -> {ref_id: list_index} for O(1) duplicate detection
    nu_reads = set()

    genomes = {}
    genomes_names = {}
    genomes_id = 0
    species_count = {}
    ref_intervals = {}
    predictions_mapping = {}

    with open(paf_path) as f:
        for line in f:
            parts = line.split()
            read_id  = parts[0]
            t_start  = int(parts[7])
            t_end    = int(parts[8])
            t_len    = int(parts[6])
            length_t = t_end - t_start
            length_q = int(parts[3]) - int(parts[2])
            nm       = int(parts[9])
            value_cig = nm / max(length_t, length_q)

            ref_prediction = parts[5]
            taxid = ref_prediction.split('|')[1]
            species_taxid = species_strain_info[taxid]
            species_count[species_taxid] = species_count.get(species_taxid, 0) + 1

            if ref_prediction not in ref_intervals:
                ref_intervals[ref_prediction] = ([], t_len)
            ref_intervals[ref_prediction][0].append((t_start, t_end))

            if ref_prediction not in genomes_names:
                genomes[genomes_id] = ref_prediction
                genomes_names[ref_prediction] = genomes_id
                genomes_id += 1
            ref_id = genomes_names[ref_prediction]

            if read_id not in read_genome_idx:
                U[read_id] = [[ref_id], [value_cig], [value_cig], value_cig]
                read_genome_idx[read_id] = {ref_id: 0}
                if mapping_class_path is not None:
                    predictions_mapping[read_id] = ref_prediction
            else:
                gidx = read_genome_idx[read_id]
                if ref_id in gidx:
                    idx = gidx[ref_id]
                    store = NU if read_id in nu_reads else U
                    if value_cig > store[read_id][1][idx]:
                        store[read_id][1][idx] = value_cig
                        store[read_id][2][idx] = value_cig
                        if value_cig > store[read_id][3]:
                            store[read_id][3] = value_cig
                else:
                    new_idx = len(gidx)
                    gidx[ref_id] = new_idx
                    if read_id not in nu_reads:
                        NU[read_id] = U.pop(read_id)
                        nu_reads.add(read_id)
                    NU[read_id][0].append(ref_id)
                    NU[read_id][1].append(value_cig)
                    NU[read_id][2].append(value_cig)
                    if value_cig > NU[read_id][3]:
                        NU[read_id][3] = value_cig
                    if mapping_class_path is not None and value_cig > NU[read_id][3]:
                        predictions_mapping[read_id] = ref_prediction

    if mapping_class_path is not None:
        mapping_output(mapping_class_path, predictions_mapping)

    for ref in ref_intervals:
        intervals, ref_len = ref_intervals[ref]
        if not intervals or ref_len == 0:
            ref_intervals[ref] = 0.0
            continue
        intervals.sort()
        cur_s, cur_e = intervals[0]
        covered = 0
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                covered += cur_e - cur_s
                cur_s, cur_e = s, e
        covered += cur_e - cur_s
        ref_intervals[ref] = covered / ref_len

    return U, NU, genomes, species_count, ref_intervals


def pathoscope_redistribution(NU):
    for j in NU:
        scores = NU[j][2]
        xsum = sum(scores)
        if xsum == 0:
            NU[j][2] = [0.0] * len(scores)
        else:
            NU[j][2] = [s / xsum for s in scores]
    return NU

def find_medoid_and_avg_distance(cluster_indices, dist_matrix):
    if len(cluster_indices) == 1:
        return cluster_indices[0], 0.0 
    
    sub_matrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
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
    references = set(genomes.values())

    for ref in references:
        taxid = ref.split('|')[1]
        species_taxid = species_strain_info[taxid]
        species_ref_dict.setdefault(species_taxid, []).append(ref)

    return species_ref_dict

def get_top_genomes(value_list, genomes):
    results = value_list[2]
    m = max(results)
    w = [i for i, x in enumerate(results) if x == m]
    p = [value_list[0][i] for i in w]
    return [genomes[i] for i in p]

def collect_species_read_data(all_mappings, genomes):
    genome_read_dict = {}
    reads_index_dict = {}
    classified = []
    reference_scores = {}
    ambigous_refs_count = {}
    ambigous_refs_reads = {}
    ambigous_species_count = 0
    ref_count = {}
    genomes_with_reads = set()

    # only allocate for genomes actually referenced by reads in this species
    relevant_genome_names = set(
        genomes[gidx]
        for vals in all_mappings.values()
        for gidx in vals[0]
    )
    n_reads = len(all_mappings)
    for name in relevant_genome_names:
        genome_read_dict[name] = [0] * n_reads
        ambigous_refs_count[name] = 0
        ambigous_refs_reads[name] = []
        ref_count[name] = 0

    for idx, read_id in enumerate(all_mappings.keys()):
        reads_index_dict[read_id] = idx

    for read_id, value_list in all_mappings.items():
        genome_list = get_top_genomes(value_list, genomes)

        if len(genome_list) == 1:
            genome = genome_list[0]
            classified.append((read_id, genome))

            reference_scores.setdefault(genome, [[], []])
            reference_scores[genome][0].append(read_id)
            reference_scores[genome][1].append(max(value_list[2]))

            ref_count[genome] += 1
            ambigous_refs_count[genome] += 1
            genome_read_dict[genome][reads_index_dict[read_id]] = 1
            genomes_with_reads.add(genome)

        else:
            for ref in genome_list:
                ambigous_refs_count[ref] += 1
                ambigous_refs_reads[ref].append(read_id)
                genome_read_dict[ref][reads_index_dict[read_id]] = 1
                genomes_with_reads.add(ref)

            ambigous_species_count += 1

    # pre-filter: only keep genomes that have at least one read assigned
    genome_read_dict = {ref: arr for ref in genomes_with_reads
                        for arr in [genome_read_dict[ref]]}

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
    for read_id, value_list in all_mappings.items():
        results_v1 = value_list[1]
        genome_list = get_top_genomes(value_list, genomes)

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
        ref_count[best_ref] += 1

        reference_scores.setdefault(best_ref, [[], []])
        reference_scores[best_ref][0].append(read_id)
        reference_scores[best_ref][1].append(max(results_v1))

    return classified, reference_scores, ref_count

def cluster_species_references(genome_read_dict, eps_value):
    # genome_read_dict is already pre-filtered by collect_species_read_data
    if len(genome_read_dict) == 0:
        return {
            "filtered_genome_read_dict": {},
            "clusters": [],
            "cluster_representatives": {},
        }

    ref_ids = list(genome_read_dict.keys())
    key_to_idx = {k: i for i, k in enumerate(ref_ids)}
    arrays = np.array(list(genome_read_dict.values()))
    dist_matrix = pairwise_distances(arrays, metric='jaccard')

    db = DBSCAN(metric='precomputed', eps=eps_value, min_samples=1)
    labels = db.fit_predict(dist_matrix)

    clusters = []
    cluster_representatives = {}

    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        medoid_idx, _ = find_medoid_and_avg_distance(cluster_indices, dist_matrix)
        medoid = ref_ids[medoid_idx]

        cluster_refs = [ref_ids[i] for i in cluster_indices]
        ordered_cluster = [medoid] + [ref for ref in cluster_refs if ref != medoid]
        clusters.append(ordered_cluster)

        for ref in cluster_refs:
            cluster_representatives[ref] = medoid

    return {
        "filtered_genome_read_dict": genome_read_dict,
        "clusters": clusters,
        "cluster_representatives": cluster_representatives,
        "ref_ids": ref_ids,
        "key_to_idx": key_to_idx,
        "dist_matrix": dist_matrix,
    }

def summarize_cluster_support(clusters, reference_scores, high_score_threshold):
    ref_high_scores = {}
    ref_high_scores_global = {}
    low_score_read_count = 0
    assigned_reads = 0

    logging.info("======Cluster support summary======")
    for i, cluster in enumerate(clusters, start=1):
        logging.info(f"Cluster {i}:")
        for ref in cluster:
            if ref in reference_scores:
                scores = reference_scores[ref]
                high_scores = [s for s in scores[1] if s >= high_score_threshold]

                low_score_read_count += sum(1 for s in scores[1] if s < high_score_threshold)
                ref_high_scores[ref] = len(high_scores)
                ref_high_scores_global[ref] = len(high_scores) / len(scores[1]) if len(scores[1]) > 0 else 0.0

                avg_score = sum(high_scores) / len(high_scores) if high_scores else 0.0
                assigned_reads += len(scores[0])

                logging.info(
                    "Reference: {}, Assigned Reads: {}, Average Score: {:.4f}, High Scores: {}".format(
                        ref, len(scores[0]), avg_score, len(high_scores)
                    )
                )
            else:
                ref_high_scores[ref] = 0
                ref_high_scores_global[ref] = 0.0
                logging.info("Reference: {}, Assigned Reads: 0, Average Score: 0.0000, High Scores: 0".format(ref))

    logging.info("Total low score reads globally: {}".format(low_score_read_count))
    logging.info("==========================================")

    return {
        "ref_high_scores": ref_high_scores,
        "ref_high_scores_global": ref_high_scores_global,
        "assigned_reads": assigned_reads,
    }


def choose_reference_reassignments(clusters, clusters_species, s_species_dist,
                                   ref_high_scores, ref_high_scores_global, cfg):
    changes = {}
    species_strong_cache = {}

    logging.info("======Reference reassignment summary======")

    for i, cluster in enumerate(clusters):
        species = clusters_species[i]
        keys, key_to_idx, dist_matrix = s_species_dist[species]

        if species not in species_strong_cache:
            strong_idx = np.array([
                idx for idx, k in enumerate(keys)
                if ref_high_scores.get(k, 0) > cfg["min_high_score_reads"]
                or ref_high_scores_global.get(k, 0.0) > cfg["min_high_score_fraction"]
            ], dtype=int)
            species_strong_cache[species] = (strong_idx, [keys[j] for j in strong_idx])
        strong_idx, strong_keys = species_strong_cache[species]

        for ref in cluster:
            if ref not in ref_high_scores:
                ref_high_scores[ref] = 0
            if ref not in ref_high_scores_global:
                ref_high_scores_global[ref] = 0.0

            weak_ref = (
                ref_high_scores[ref] < cfg["min_high_score_reads"] or
                (
                    ref_high_scores[ref] < cfg["mid_high_score_reads"] and
                    ref_high_scores_global[ref] < cfg["min_high_score_fraction"]
                )
            )

            if not weak_ref:
                continue

            ref_i = key_to_idx[ref]
            for j in np.argsort(dist_matrix[ref_i, strong_idx]):
                cand = strong_keys[j]
                if cand == ref or cand in changes:
                    continue
                changes[ref] = cand
                logging.info(
                    "Cluster {} - Reference {} changed to {} based on clustering with distance {:.4f}".format(
                        i + 1, ref, cand, dist_matrix[ref_i, strong_idx[j]]
                    )
                )
                break

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
        max_ref = max(cluster, key=lambda r: ref_count.get(r, 0))

        if ref_count.get(max_ref, 0) > 0:
            for ref in cluster:
                changes[ref] = max_ref
            ordered = [max_ref] + [ref for ref in cluster if ref != max_ref]
        else:
            ordered = cluster

        representatives_new.append(ordered[0])
        updated_clusters.append(ordered)

    return changes, representatives_new, updated_clusters

def write_final_cluster_outputs(clustering_out, updated_clusters, representatives_new):
    with open(os.path.join(clustering_out, "clusters.txt"), "w") as f_clusters:
        for cluster in updated_clusters:
            f_clusters.write(" ".join(cluster).strip() + "\n")

    with open(os.path.join(clustering_out, "representatives.txt"), "w") as f_rep:
        for rep in representatives_new:
            f_rep.write(rep + "\n")

def write_final_assignments(classified, changes, cluster_changes, output_path):
    with open(output_path, "w") as f:
        for read_id, ref in classified:
            ref = changes.get(ref, ref)
            final_ref = cluster_changes.get(ref, ref)
            f.write(f"{read_id} : {final_ref}\n")

def write_reference_summary(classified, changes, cluster_changes, ref_intervals, output_path):
    """
    Write a TSV with one row per final reference:
        reference_name  \t  read_count  \t  horizontal_coverage
    """
    ref_count = {}
    for read_id, ref in classified:
        ref = changes.get(ref, ref)
        final_ref = cluster_changes.get(ref, ref)
        ref_count[final_ref] = ref_count.get(final_ref, 0) + 1

    with open(output_path, "w") as f:
        f.write("reference\tread_count\thorizontal_coverage\n")
        for ref, count in sorted(ref_count.items(), key=lambda x: x[1], reverse=True):
            hcov = compute_horizontal_coverage(ref_intervals, ref)
            f.write(f"{ref}\t{count}\t{hcov:.6f}\n")

    logging.info(f"Reference summary written to {output_path}")


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
    U, NU, genomes, species_count, ref_intervals = calculate_mapping_class(
        args.paf_path, species_strain_info, mapping_class_path=None, beta=0.5
    )

    species_class = species_identification_with_thresholds(
        U, NU, genomes, species_count, args.strain_species_info,
        min_read_count=cfg["species_min_read_count"],
        min_mean_score=cfg["species_min_mean_score"],
        low_count_cap=cfg["species_low_count_cap"]
    )
    U_species, NU_species = species_split(U, NU, species_class, genomes, species_strain_info)

    species_ref_dict = build_species_ref_dict(genomes, species_strain_info)
    species = list(set(species_class.values()))

    s_species_dist = {}
    clusters = []
    clusters_species = []
    classified = []
    reference_scores = {}

    for s in species:
        logging.info("Species {} has {} references, eps value for clustering: {}".format(
            s, len(species_ref_dict[s]), cfg["cluster_eps"])
        )

        NU_result = pathoscope_redistribution(NU_species[s])
        all_mappings = {**U_species[s], **NU_result}

        species_data = collect_species_read_data(all_mappings, genomes)

        classified.extend(species_data["classified"])
        reference_scores.update(species_data["reference_scores"])

        classified, reference_scores, _ = resolve_ambiguous_reads(
            all_mappings,
            genomes,
            species_data["ambigous_refs_count"],
            classified,
            reference_scores,
            species_data["ref_count"]
        )

        clustering_result = cluster_species_references(
            species_data["genome_read_dict"],
            cfg["cluster_eps"]
        )

        if clustering_result["clusters"]:
            s_species_dist[s] = (
                clustering_result["ref_ids"],
                clustering_result["key_to_idx"],
                clustering_result["dist_matrix"],
            )
        clusters.extend(clustering_result["clusters"])
        clusters_species.extend([s] * len(clustering_result["clusters"]))

    support_summary = summarize_cluster_support(
        clusters, reference_scores, cfg["high_score_threshold"]
    )
    changes = choose_reference_reassignments(
        clusters,
        clusters_species,
        s_species_dist,
        support_summary["ref_high_scores"],
        support_summary["ref_high_scores_global"],
        cfg,
    )

    ref_count = get_ref_count(classified, changes)

    changes2, representatives_new, updated_clusters = recompute_cluster_representatives_by_count(
        clusters, ref_count
    )

    active = [ref_count.get(rep, 0) > 0 for rep in representatives_new]
    updated_clusters    = [c for c, keep in zip(updated_clusters,    active) if keep]
    representatives_new = [r for r, keep in zip(representatives_new, active) if keep]

    write_final_cluster_outputs(args.clustering_out, updated_clusters, representatives_new)
    write_final_assignments(classified, changes, changes2, args.read_class_output)
    write_reference_summary(classified, changes, changes2, ref_intervals, args.ref_summary_output)

    logging.info("Total assigned reads: {}".format(len(classified)))


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
        help="Minimum read count threshold for species reliability (default=5)."
    )

    parser.add_argument(
        "--species_min_mean_score", type=float, default=0.6,
        help="Minimum mean score threshold for species reliability (default=0.6)."
    )

    parser.add_argument(
        "--species_low_count_cap", type=int, default=30,
        help="Species with mean score below threshold and read count below this cap are treated as unreliable (default=30)."
    )

    parser.add_argument(
        "--read_class_output", type=str, default="read_classification.out",
        help="Path to the output file with classification labels for reads (default=read_classification.out)."
    )

    parser.add_argument(
        "--clustering_out", type=str, default="clustering_output",
        help="Path to the directory for clustering-related output files (default=clustering_output)."
    )

    parser.add_argument(
        "--ref_summary_output", type=str, default="reference_summary.tsv",
        help="Path to TSV file with per-reference read counts and horizontal coverage (default=reference_summary.tsv)."
    )

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()