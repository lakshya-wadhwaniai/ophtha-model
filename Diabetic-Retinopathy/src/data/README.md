This directory contains scripts for one-time preprocessing of raw data to prepare it for training machine learning models

#### 1. Download raw data from CVAT annotation tool

```bash
$ python download.py --path /path/to/save/dataset/ --start start_task_id --end final_task_id
```

- Currently `start_task_id` is 1 and `final_task_id` is 358.
- This script downloads data from CVAT annotation tool corresponding to the range from `start_task_id` to `final_task_id` (both inclusive).
- Get the annotation `task_id` from the annotation job created.
- We download YOLO-style annotations. Checkout [CVAT APIs](https://wadhwaniai.atlassian.net/wiki/spaces/TU/pages/650117121/Annotation) to download annotations in other formats.
- Currently this raw data is stored at `/scratchg/data/tb-ultrasound/cvat_annotations/cvat_annot_1`


#### 2. Extract downloaded raw data and structure it as per annotator id

```bash
$ python make_annotated_data.py --source /path/to/downloaded/raw/data --target /target/path
```

- `source` is the path of the raw data from previous step
- `target` is the desired path for extraced data
- Current `target` is `/scratchg/data/tb-ultrasound/cvat_annotations/task_annotations`

##### NOTE:

- This script is NOT generalizable
- It assumes a certain order in which annotation job was created. 
- For each video, there are 2 annotation jobs since there are two annoatators Dr.Ajit and Dr.Onkar
- E.g. for 2 videos there are 4 jobs

  Job 1 -> Dr_Onkar_task_1
  
  Job 2 -> Dr_Ajit_task_1
  
  Job 3 -> Dr_Onkar_task_2
  
  Job 4 -> Dr_Ajit_task_2 

- The tasks are not mapped to the video id. So $i^{th}$ task doesn't necessarily correspond to $i^{th}$ video. The mapping between task number and video number is not defined as of now.
- For the current set of data, we had to resort to a hack to get this mapping. Next section describes about it.
- For the next round of annotation, we recommend doing a one-to-one mapping between task and video ID


#### 3. Split the entire extracted data into positive and negative and preprocess the frames

```bash
$ python data_prep_preprocess.py
```

- This takes the source folder - task_annotations and arranges the data in the format divided according to patients, videos and annotators. It also crops images and makes adjustment to the bounding box annotations.
- This script internally calls `./adjust_bbox.py`
- The processed data root is at `/scratchg/data/tb-ultrasound/fs2_data/`
- The structure of the data stored is `path/to/data/root/{positive,negative}/{patientID}/{videoID}/{annotator}/{frame1.jpg}`
- The corresponding bounding box is available at `/path/to/data/root/{negative,positive}/{patientID}/{videoID}/{annotator}/{frame1.txt}`, but however it is not required for classification task.

##### NOTE:

- This script is NOT generalizable
- The split into positive and negative is done based on video ID which may not be same in future. For positive videos, it is NP00x and for negative it is N00x. So if there's a 'P' in the ID, this implies it is positive.
- Also the mapping between `task_id` and `video_id` is hacky and hardcoded. It is stored at `/scratchg/data/tb-ultrasound/cvat_annotations/task_video_map.txt`
- In preprocessing, we remove text artifacts from frames such as patient name, lab details, etc by cropping. Cropping happens at https://github.com/WadhwaniAI/ultrasound/blob/master/src/dataset/data_prep_and_preprocess.py#L56
- Again, this is specific to this particular round of data collection. The next round may be collected from another USG machine which may or may not have these artifacts. The position of artifacts could also be different. So this step needs to be re-visited.

#### 4. Generate labels for classification

```bash
$ python create_labels.py
```

- generates a json file with list of all images and corresponding labels
```py
{
    # image_path : label
    "/scratchg/data/tb-ultrasound/fs2_data/positive/NP011/video2/Dr_Ajit_task_no_1/207.jpg": 0
}
```
- current file is stored at `/scratchg/data/tb-ultrasound/fs2_data/splits/labels.json`
- requires that the parent folder (`/scratchg/data/tb-ultrasound/fs2_data/splits/`) is already created


#### 5. Create split

The split is created based on patient and NOT on video or frames because we need evaluation at a patient level and doing it at video or frame might lead to data leakage. The strategy we adopt to split is based on prevelance of positive frames.

- Run the `notebooks/data_analysis.ipynb` notebook to create the split. Run it iteratively unless you get a random split with almost equal prevelance in all three sets (train, test, val).
- The notebook generates `patient_split.yaml` which contains only metadata info for split.
- Run `create_splits.py` to generate separate frame level split files for test, train and val. This create splits at `/path/to/split/root/{version}/{train,test,val}.txt`
- The splits are versioned.
- Currently used stable version is `v3`
- The current split is `/scratchg/data/tb-ultrasound/fs2_data/splits/v3`

#### 6. Create dataset metadata

Run `notebooks/metadata.ipynb` to create metadata about dataset required for evaluation

