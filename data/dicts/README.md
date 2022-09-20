This directory hosts files containing the serialized data dictionaries which can optionally be used when executing the algorithms OpenMax and EVM. The main idea is to avoid a full data extraction procedure upon every script execution (-> lower execution time, ideal during development or for debugging).

Note: Because of their storage requirements, these files are not uploaded to GitLab. In fact, they are optional since using them simply speeds up the script execution time. Furthermore, they are automatically created when configuring a run of the mentioned algorithms.
