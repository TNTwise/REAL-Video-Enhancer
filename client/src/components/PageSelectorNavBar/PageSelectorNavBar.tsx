import { Image, HStack, Link as ChakraLink } from "@chakra-ui/react";
import { Link as RouterLink } from "react-router-dom";
import {
  primaryBGColor,
  primaryWidgetColor,
  secondaryWidgetColor,
} from "@/styles/styles";
import homeIcon from "@/assets/icons/home.svg";
import downloadIcon from "@/assets/icons/download.svg";

const links = [
  { name: "Home", href: "/home", image: homeIcon },
  { name: "Download", href: "/download", image: downloadIcon },
];

export const Navbar = () => {
  return (
    <HStack
      gap={8}
      align="center"
      bg={primaryBGColor}
      padding="10px"
      borderRadius="10px"
      marginBottom="10px"
    >
      {links.map((link) => (
        <ChakraLink
          key={link.name}
          as={RouterLink}
          to={link.href} // ignore
          fontWeight="medium"
          transition="color 0.2s ease"
        >
          <Image
            background={primaryWidgetColor}
            padding="5px"
            borderRadius="5px"
            _hover={{
              background: secondaryWidgetColor,
            }}
            transition="color 2s ease"
            src={link.image}
            w="50px"
            h="50px"
          ></Image>
        </ChakraLink>
      ))}
    </HStack>
  );
};
