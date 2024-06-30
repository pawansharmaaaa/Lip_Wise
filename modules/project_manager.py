import os

LIPWISE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA = {}

class Project:
    def __init__(self, project_name, specified_dir=None):
        self.parent_dir = os.path.join(LIPWISE_DIRECTORY, 'Projects')
        self.project_name = project_name
        
        if specified_dir:
            self.project_dir = os.path.join(specified_dir, self.project_name)
        else:
            self.project_dir = os.path.join(self.parent_dir, self.project_name)

        PROJECT_DATA[self.project_name] = {}
    
    def create_project(self):
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
        else:
            print('Project already exists')

class Person:
    def __init__(self, person_name: str, project: Project):
        self.person_name = person_name

        self.person_dir = os.path.join(project.project_dir, person_name)

        self.npy_files_dir = os.path.join(self.person_dir, 'npy_files')
        self.media_dir = os.path.join(self.person_dir, 'media')
        
        if not os.path.exists(self.person_dir):
            os.makedirs(self.person_dir)
        if not os.path.exists(self.npy_files_dir):
            os.makedirs(self.npy_files_dir)
        if not os.path.exists(self.media_dir):
            os.makedirs(self.media_dir)

        self.bbox = os.path.join(self.person_dir, 'bbox.npy')
        self.presence_mask = os.path.join(self.person_dir, 'presence_mask.npy')
        self.kps = os.path.join(self.person_dir, 'kps.npy')
        self.landmarks = os.path.join(self.person_dir, 'landmarks.npy')

        self.preview_video_path = os.path.join(self.person_dir, 'preview.mp4')
        self.output_video_path = os.path.join(self.person_dir, 'output.mp4')
    
    def save_data(self):
        PROJECT_DATA[self.person_name] = {
            'bbox': self.bbox,
            'presence_mask': self.presence_mask,
            'kps': self.kps,
            'landmarks': self.landmarks,
            'preview_video_path': self.preview_video_path,
            'output_video_path': self.output_video_path
        }

def main():
    project = Project('Test_Project')
    project.create_project()

    person = Person('Pawan', project)

    print(person.person_dir)
    print(person.npy_files_dir)
    print(person.media_dir)
    print(person.bbox)
    print(person.presence_mask)
    print(person.kps)
    print(person.landmarks)
    print(person.preview_video_path)
    print(person.output_video_path)

if __name__ == '__main__':
    main()