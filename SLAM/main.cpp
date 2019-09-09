#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <System.h>

using namespace vizdoom;
using namespace std;

int main(){

    std::cout << "\n\nSPECTATOR EXAMPLE\n\n";


    DoomGame *game = new DoomGame();

    // Choose scenario config file you wish to be watched by agent.
    // Don't load two configs cause the second will overwrite the first one.
    // Multiple config files are ok but combining these ones doesn't make much sense.

    game->loadConfig("/home/ruilong/CLionProjects/Doom_test/scenarios/my_way_home.cfg");
    std::cout << "\n\nconfig loaded\n\n";
    //game->loadConfig("../../scenarios/deadly_corridor.cfg");
    //game->loadConfig("../../scenarios/deathmatch.cfg");
    //game->loadConfig("../../scenarios/defend_the_center.cfg");
    //game->loadConfig("../../scenarios/defend_the_line.cfg");
    //game->loadConfig("../../scenarios/health_gathering.cfg");
    //game->loadConfig("../../scenarios/my_way_home.cfg");
    //game->loadConfig("../../scenarios/predict_position.cfg");
    //game->loadConfig("../../scenarios/take_cover.cfg");

    game->setDoomGamePath("/home/ruilong/CLionProjects/Doom_test/scenarios/freedoom2.wad");
    std::cout << "\n\nwad loaded\n\n";
    //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

//    game->setScreenResolution(RES_640X480);

    // Enables spectator mode, so You can play and agent watch your actions.
    // You can only use the buttons selected as available.
    game->setMode(SPECTATOR);
//    void setTicrate(unsigned int 20);
    game->setTicrate(20);

    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);


    game->init();
    game->sendGameCommand("movebob 0.0");
    string path_Vocabulary = "/home/ruilong/CLionProjects/Doom_test/Vocabulary/ORBvoc.txt";
    string path_Setting = "/home/ruilong/CLionProjects/Doom_test/TUM1.yaml";
    ORB_SLAM2::System SLAM(path_Vocabulary,path_Setting,ORB_SLAM2::System::MONOCULAR,true);
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    // Run this many episodes
    int episodes = 10;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            int timeStamps=0;
            timeStamps++;
            cv::Mat frame(600, 800, CV_8UC3);

            // Get the state.
            GameStatePtr state = game->getState();
            BufferPtr screenBuf = state->screenBuffer;

            for (int k = 0; k < frame.rows; ++k) {
                for (int j = 0; j < frame.cols; ++j) {
                    auto vectorCoord = 3 * (k * frame.cols + j);
                    // Mapping from image coordinates to coordinates in vector (array)

                    frame.at<uchar>(k, 3 * j) = (*screenBuf)[vectorCoord + 2];
                    frame.at<uchar>(k, 3 * j + 1) = (*screenBuf)[vectorCoord + 1];
                    frame.at<uchar>(k, 3 * j + 2) = (*screenBuf)[vectorCoord];
                }
            }


            SLAM.TrackMonocular(frame, timeStamps);

            // Advances action - lets You play next game tic.
            game->advanceAction();

            // You can also advance a few tics at once.
            // game->advanceAction(4);

            // Get the last action performed by You.
            std::vector<double> lastAction = game->getLastAction();

            // And reward You get.
            double reward = game->getLastReward();

            std::cout << "State #" << state->number << "\n";
            std::cout << "Action made:";
            for(auto a: lastAction) std::cout << " " << a;
            std::cout << "\n";
            std::cout << "Action reward: " << reward <<"\n";
            std::cout << "=====================\n";

        }

        std::cout << "Episode finished.\n";
        std::cout << "Total reward: " << game->getTotalReward() << "\n";
        std::cout << "************************\n";
    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    SLAM.Shutdown();
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    delete game;
}