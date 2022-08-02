#######################################################################################################################
# Example of extraction of annotations from ELAN annotation files
#######################################################################################################################

import glob
import pympi


# ELAN annotation tier names
eaf_tier_names = [
	'mouth_open',

	'food_to_mouth',
	'food_entered',
	'food_lifted',

	'drink_to_mouth',
	'drink_entered',
	'drink_lifted',

	'napkin_to_mouth',
	'napkin_entered',
	'napkin_lifted',

	'disruption'
]

annotations_folder = f'/home/aa2375/social-dining/data/annotation/annotation-files'
annotation_files = sorted(glob.glob(f'{annotations_folder}/*.eaf'))
print(annotation_files)
# For all annotation files
for i in range(len(annotation_files)):

	eaf_obj = pympi.Elan.Eaf(annotation_files[i])

	for tier_name in eaf_tier_names:
		assert tier_name in eaf_obj.get_tier_names(), f"WARNING: no '{tier_name}' annotations found for: {annotation_files[i]}"

		# times are in milliseconds
		for (start_time, end_time, value) in eaf_obj.get_annotation_data_for_tier(tier_name):
			print(start_time, end_time, value)
