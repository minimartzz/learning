# DVC Notes

DVC is used in tandem with `git` to manage and version control datasets

## Tracking datasets

1. `dvc init`: on an existing git repo to start tracking
2. `dvc add <file>`: adds an existing file to be tracked
3. `git add <file>.dvc .gitignore`: these 2 files need to be tracked. The `.dvc` file is the metadata for the raw data and `.gitignore` prevents git from storing the original data file
4. `dvc remote add -d <datastore name> <path/to/datastore>`: path to data store can be a local folder, shared drive or cloud storage
5. `dvc push`: pushes the data to the new store

🚨 Important Files 🚨

- `.dvc` folder contains all the necessary tracking info (similar to `.git` folder)
- `<file>.xml.dvc` contains the metadata needed to validate and retrieve data

## Pulling data

Assuming that the datastore is correctly specified in the metadata.

`dvc pull` will retrieve the data from the datastore

## Switching between versions

Switching between data versions involves checking out specific commits of the `<file>.xml.dvc` file.

```bash
git checkout HEAD~1 data/data.xml.dvc
# or
git checkout <commit-hash> data/data.xml.dvc
```

## Using Minio S3 as Datastore

_Reference: https://medium.com/@murisuu/setting-up-data-version-control-dvc-with-minio-a-complete-guide-e3d8d8a8b07a_

Adding files

1. `dvc remote add -d minio s3://dvcstore`: define the remote data store as the s3 bucket name
2. Add the following configurations

```bash
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id <MINIO_USER>
dvc remote modify --local minio secret_access_key <MINIO_PASSWORD>
```

3. `dvc push`: push tracked files

Removing files

1. `dvc remove <file>.dvc [--outs]`: removes the metadata file tracker. Optionally `--outs` flag removes local instance of data
2. `dvc gc --workspace --cloud`: `--workspace` cleans up cache in working folder. `--cloud` removes files from minio store

⚠️ Known Issues

- Sometimes if minio is not accessible from `localhost:9000` using the ipv6 loopback address as the **endpointurl** might work `http://[::1]:9000`

## Data registries

Data registries are DVC metadata files that point to remote data stores (e.g s3 buckets). They allow users to pull data directly from the git repo using the metadata information

Prerequisites:

1. Know where the remote storage path is and have it added e.g `dvc remote add -d <datastore name> <path/to/datastore>`
2. Have the necessary credentials and permissions to access the data store

Commands

```bash
# List files in the repository
dvc list -R https://github.com/iterative/dataset-registry

# Get data files: <repo> <folder/file>
dvc get https://github.com/example/registry music/songs

# Import - different from get in that you get DVC metadata too
dvc import https://github.com/example/registry images/faces

# Updates dvc files
dvc update <file>.dvc
```
