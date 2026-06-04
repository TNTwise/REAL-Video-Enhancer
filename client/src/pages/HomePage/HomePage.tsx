import { Center, Image, Text } from "@chakra-ui/react";
import RVELogo from '/src/assets/logo-v2.svg';

export function HomePage () {
    return (
        <Center>
            <Image src={RVELogo} w="150px" h="150px"></Image>
            <Text padding="15px" >REAL Video Enhancer</Text>
        </Center>
    );
}